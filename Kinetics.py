import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter, defaultdict
from scipy.stats import linregress
from scipy.constants import R  # 8.314 J/mol·K


molecule_configs = {
    "cyclo": {
        "smiles": "O=C1CCCCC1",
        "formation_smiles": ["C=C", "C=C=O", "C#C"],
        "initial_count": 16,
        "folder": "./reaction events cyclo - Kinetics/",
        "pattern": "reaction_events_*.csv"
    },
    "VG": {
        "smiles": "OCC(O)CO",
        "formation_smiles": ["C=O","[CH2][CH]O"],
        "initial_count": 10,
        "folder": "./VG-reaction events/",
        "pattern": "reaction_events_*.csv"
    }
}

print("Available molecule keys:", ", ".join(molecule_configs.keys()))
selected_key = input("Enter target molecule key: ").strip()
if selected_key not in molecule_configs:
    raise ValueError(
        f"Invalid key '{selected_key}'. Please use one of: {', '.join(molecule_configs.keys())}"
    )

cfg = molecule_configs[selected_key]
target_smiles = cfg["smiles"]
formation_targets = cfg.get("formation_smiles")
if isinstance(formation_targets, str):
    formation_targets = [formation_targets]
initial_count = cfg["initial_count"]
pattern = os.path.join(cfg["folder"], cfg["pattern"])

print(f"\nUsing target: {selected_key}")
print(f"  Consumption target: {target_smiles}")
print(f"  Formation targets:  {formation_targets if formation_targets else 'None'}")
print(f"  Initial count:      {initial_count}")
print(f"  Folder:             {cfg['folder']}\n")


def extract_k(file_path, formation_list=None):
    df = pd.read_csv(file_path)

    required_cols = {"Reactants", "Products", "Initial time (fs)"}
    if not required_cols.issubset(df.columns):
        return None, {}

    df["Reactants"] = df["Reactants"].astype(str)
    df["Products"] = df["Products"].astype(str)
    df = df.dropna(subset=["Initial time (fs)"])

    mask_c = df["Reactants"].str.contains(target_smiles, regex=False, na=False)
    consumed_times = (df.loc[mask_c, "Initial time (fs)"] * 1e-3).tolist()  # ps
    consumed_counts = Counter(consumed_times)

    count_c = initial_count
    t_c, N_c = [0.0], [count_c]

    for t in sorted(consumed_counts.keys()):
        count_c -= consumed_counts[t]
        count_c = max(0, count_c)
        t_c.append(t)
        N_c.append(count_c)

    t_c = np.array(t_c, dtype=float)
    N_c = np.array(N_c, dtype=float)

    k_cons = None
    if len(t_c) > 2 and np.any(N_c > 0):
        lnNc = np.log(N_c[N_c > 0])
        tc = t_c[N_c > 0]
        slope, _, _, _, _ = linregress(tc, lnNc)
        k_cons = -slope  # ps^-1

    k_forms = {}
    if formation_list:
        products_per_row = [
            [p.strip() for p in str(val).split("+")] if pd.notna(val) else []
            for val in df["Products"].values
        ]

        times_ps = (df["Initial time (fs)"].values * 1e-3).tolist()  # ps

        for prod in formation_list:
            times_hit = []
            for prods, t_ps in zip(products_per_row, times_ps):
                if prod in prods:
                    times_hit.append(t_ps)

            counts = Counter(times_hit)

            nf = 0
            tf_seq, Nf_seq = [0.0], [0]

            for t in sorted(counts.keys()):
                nf += counts[t]
                tf_seq.append(t)
                Nf_seq.append(nf)

            tf_seq = np.array(tf_seq, dtype=float)
            Nf_seq = np.array(Nf_seq, dtype=float)

            kf = None
            if len(tf_seq) > 2 and np.any(Nf_seq > 0):
                slope, _, _, _, _ = linregress(tf_seq, Nf_seq)
                kf = slope / initial_count  # ps^-1

            k_forms[prod] = kf

    return k_cons, k_forms


rep_cons = defaultdict(list)
rep_form = defaultdict(lambda: defaultdict(list))

for path in glob(pattern):
    fname = os.path.basename(path)

    # expected format like reaction_events_2250-1.csv
    m = re.search(r"_(\d+)-\d+\.csv$", fname)
    if not m:
        continue

    T = int(m.group(1))
    k_cons, k_forms = extract_k(path, formation_targets)

    if k_cons is not None:
        rep_cons[T].append(k_cons)
        print(f"✓ {fname} → k_cons = {k_cons:.2e} ps⁻¹")

    if formation_targets:
        for prod, kf in k_forms.items():
            if kf is not None:
                rep_form[prod][T].append(kf)
                print(f"  ↳ {prod}: k_form = {kf:.2e} ps⁻¹")

print()


def fit_arrhenius(temps, ks):
    temps = np.array(temps, dtype=float)
    ks = np.array(ks, dtype=float)

    invT = 1000.0 / temps
    lnk = np.log(ks)

    slope, intercept, r_value, _, _ = linregress(invT, lnk)
    Ea = -slope * R * 1000.0   # J/mol
    A0 = np.exp(intercept)     # ps^-1

    return Ea, A0, r_value**2, invT, lnk, slope, intercept


def plot_and_report(label, temps, ks, save_as=None):
    Ea, A0, r2, invT, lnk, m, b = fit_arrhenius(temps, ks)

    plt.figure(figsize=(7, 5))
    plt.plot(invT, lnk, "o")
    plt.plot(invT, m * invT + b, label=f"Ea={Ea/4184:.1f} kcal/mol\nA₀={A0:.1e} ps⁻¹\nR² = {r2:.4f}\n")
    plt.xlabel("1000 / T (1/K)")
    plt.ylabel("ln(k) (ps⁻¹)")
    plt.title("Arrhenius Plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=200)

    plt.show()

    print(f"{label} Arrhenius parameters:")
    print(f"  Ea = {Ea:.2e} J/mol = {Ea/1000:.2f} kJ/mol = {Ea/4184:.2f} kcal/mol")
    print(f"  A₀ = {A0:.2e} ps⁻¹ = {A0*1e12:.2e} s⁻¹ ; Ln(Ao)= {np.log(A0*1e12)}")
    print(f"  R² = {r2:.4f}\n")

if len(rep_cons) >= 2:
    T_cons = np.array(sorted(rep_cons.keys()), dtype=float)
    k_cons_avg = np.array([np.mean(rep_cons[T]) for T in T_cons], dtype=float)
    plot_and_report("Consumption", T_cons, k_cons_avg)
else:
    print("Not enough data for Consumption Arrhenius plot.\n")

if formation_targets:
    for prod in formation_targets:
        data = rep_form[prod]
        if len(data) < 2:
            print(f"Not enough data for Formation ({prod})")
            continue

        T_form = np.array(sorted(data.keys()), dtype=float)
        k_form_avg = np.array([np.mean(data[T]) for T in T_form], dtype=float)
        plot_and_report(f"Formation [{prod}]", T_form, k_form_avg)