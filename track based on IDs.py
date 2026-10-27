import os
import glob
import ast
import pandas as pd
import networkx as nx
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdChemReactions
from PIL import Image, ImageDraw, ImageFont

RDLogger.DisableLog('rdApp.*')

input_folder = "./Anealing- reaction_events_species_based"
feedstock = "O=C1CCCCC1"   # cyclohexanone (VG when running validation or any other feedstock later)
main_products = {"C=C", "C=C=O", "C#C", "[CH2][CH2]"}     # Main products found in the simulations

os.makedirs("./Annealing- pathway_images-check/", exist_ok=True)  #All detected paths from all files
os.makedirs("./Annealing- pathway_logs-check/", exist_ok=True)

ignore_small = {"[OH]", "O", "[O]", "[H]", "[H][H]", "[HO]"}


def safe_literal_eval(x):
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []


def safe_canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def extract_main_reactant_id(reactant_ids, id2smi_raw):
    """
    Select one main reactant ID for graph traversal

    """
    for rid in reactant_ids:
        smi = id2smi_raw.get(rid, None)
        if smi is not None and smi not in {"[OH]", "O", "[O]", "[H]", "[H][H]", "[HO]"}:  #This doesn't mean that small species will not be considered, but they will not be tracked
            return rid
    return reactant_ids[0] if reactant_ids else None


def get_species_smiles_list(ids, id2smi):
    smis = []
    for sid in ids:
        smi = id2smi.get(sid, None)
        if smi is not None:
            smis.append(smi)
    return smis


def normalize_species_list(smis):

    out = []
    for smi in smis:
        cs = safe_canonical_smiles(smi)
        if cs is not None:
            out.append(cs)
    return tuple(sorted(out))


def first_step_is_feedstock_only(path, G, cf):
    """
     Keep only pathways whose first step begins with exactly one unique reactant(case dependent - Unimolecular decomposition)
     This can be omitted for other cases
     """
    if len(path) < 2:
        return False

    u, v = path[0], path[1]
    edge = G[u][v]

    reactant_smis = edge.get("reactant_smis", [])
    reactant_smis = [s for s in reactant_smis if s is not None]

    canon = []
    seen = set()
    for smi in reactant_smis:
        cs = safe_canonical_smiles(smi)
        if cs is not None and cs not in seen:
            canon.append(cs)
            seen.add(cs)

    return len(canon) == 1 and canon[0] == cf


def path_has_reverse_reaction(path, G):

    seen_steps = []

    for u, v in zip(path[:-1], path[1:]):
        edge = G[u][v]

        r = normalize_species_list(edge.get("reactant_smis", []))
        p = normalize_species_list(edge.get("product_smis", []))

        if (p, r) in seen_steps:
            return True

        seen_steps.append((r, p))

    return False


def draw_step(reactant_smis, product_smis, step_idx):
    rxn = rdChemReactions.ChemicalReaction()

    valid_reactants = []
    valid_products = []

    for smi in reactant_smis:
        if smi is None:
            continue
        smi = str(smi).strip()
        if not smi:
            continue
        try:
            rm = Chem.MolFromSmiles(smi)
            if rm is not None:
                rxn.AddReactantTemplate(rm)
                valid_reactants.append(smi)
            else:
                print(f"Step {step_idx} Invalid reactant SMILES: {smi}")
        except Exception as e:
            print(f"Step {step_idx} Reactant parse error: {smi}")

    for smi in product_smis:
        if smi is None:
            continue
        smi = str(smi).strip()
        if not smi:
            continue
        try:
            pm = Chem.MolFromSmiles(smi)
            if pm is not None:
                rxn.AddProductTemplate(pm)
                valid_products.append(smi)
            else:
                print(f"Step {step_idx} Invalid product SMILES: {smi}")
        except Exception as e:
            print(f" Product parse error")

    if valid_reactants or valid_products:
        core = Draw.ReactionToImage(rxn, subImgSize=(350, 250))
    else:
        core = Image.new("RGB", (700, 250), "white")
        d0 = ImageDraw.Draw(core)
        f0 = ImageFont.load_default()
        d0.text((10, 10), f"Could not render Step {step_idx}", fill=(0, 0, 0), font=f0)

    caption = f"Step {step_idx}: {','.join(reactant_smis)} → {','.join(product_smis)}"
    font = ImageFont.load_default()

    img = Image.new("RGB", (core.size[0], core.size[1] + 40), "white")
    img.paste(core, (0, 0))
    d = ImageDraw.Draw(img)
    d.text((5, core.size[1] + 5), caption, fill=(0, 0, 0), font=font)

    return img


def vstack(images):
    if not images:
        return Image.new("RGB", (800, 200), "white")

    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]

    new = Image.new("RGB", (max(widths), sum(heights)), "white")

    y = 0
    for im in images:
        new.paste(im, (0, y))
        y += im.size[1]

    return new


cf = safe_canonical_smiles(feedstock)
cprod = {safe_canonical_smiles(s) for s in main_products}
cprod = {s for s in cprod if s is not None}

files = glob.glob(os.path.join(input_folder, "*_speciesIDs.csv"))

for f in files:
    base = os.path.basename(f)
    print(f"\n Processing {base} ")  ##

    df = pd.read_csv(f)
    species_map_file = f.replace("_speciesIDs.csv", "_species_map.csv")

    if not os.path.exists(species_map_file):
        print(f"Missing species map file for {base}")
        continue

    species_map = pd.read_csv(species_map_file)

    if "Molecule_ID" not in species_map.columns or "Canonical_SMILES" not in species_map.columns:
        print(f"species_map missing required columns in {species_map_file}")
        continue

    species_map["canonical"] = species_map["Canonical_SMILES"].apply(safe_canonical_smiles)

    id2smi_raw = dict(zip(species_map["Molecule_ID"], species_map["Canonical_SMILES"]))
    id2smi = dict(zip(species_map["Molecule_ID"], species_map["canonical"]))

    start_nodes = species_map[species_map["canonical"] == cf]["Molecule_ID"].tolist()
    if not start_nodes:
        print("Feedstock not found in this dataset.")
        continue

    target_nodes = species_map[species_map["canonical"].isin(cprod)]["Molecule_ID"].tolist()
    if not target_nodes:
        print("No target products in this dataset.")
        continue

    G = nx.DiGraph()

    for _, row in df.iterrows():
        rxn = row.get("Reaction_HYBRID", None)
        reactant_ids = safe_literal_eval(row.get("Reactant_Species_IDs", "[]"))
        product_ids = safe_literal_eval(row.get("Product_Species_IDs", "[]"))

        if not reactant_ids or not product_ids:
            continue

        main_rid = extract_main_reactant_id(reactant_ids, id2smi_raw)
        if main_rid is None:
            continue

        main_r_smi = id2smi.get(main_rid, None)
        if main_r_smi is None:
            continue

        reactant_smis = get_species_smiles_list(reactant_ids, id2smi)
        product_smis = get_species_smiles_list(product_ids, id2smi)

        for pid in product_ids:
            pid_smi = id2smi.get(pid, None)
            if pid_smi is None:
                continue

            G.add_edge(
                main_rid,
                pid,
                main_reactant_id=main_rid,
                main_reactant_smi=main_r_smi,
                reactant_smis=reactant_smis,
                product_smis=product_smis,
                rxn=rxn
            )

    all_paths = []
    MAX_STEPS = 4
    for s in start_nodes:
        for t in target_nodes:
            if s in G and t in G and nx.has_path(G, s, t):
                for path in nx.all_simple_paths(G, s, t, cutoff=MAX_STEPS):
                    if not first_step_is_feedstock_only(path, G, cf):
                        continue
                    if path_has_reverse_reaction(path, G):
                        continue
                    all_paths.append(path)

    if not all_paths:
        print("No pathways found after filtering.")
        continue

    for pi, path in enumerate(all_paths, 1):
        images = []
        logs = [f"Pathway {pi} from {base}"]

        for step_idx, (u, v) in enumerate(zip(path[:-1], path[1:]), 1):
            edge = G[u][v]

            reactant_smis = edge.get("reactant_smis", [])
            product_smis = edge.get("product_smis", [])

            print(f"\nStep {step_idx}")
            print("Reactants:", reactant_smis)
            print("Products :", product_smis)

            img = draw_step(reactant_smis, product_smis, step_idx)
            images.append(img)

            logs.append(f"Step {step_idx}: {','.join(reactant_smis)} → {','.join(product_smis)}")

        final = vstack(images)

        img_out = f"./Annealing- pathway_images/{base}_path_{pi}.png"
        log_out = f"./Annealing- pathway_logs/{base}_path_{pi}.txt"

        final.save(img_out)
        with open(log_out, "w", encoding="utf-8") as ftxt:
            ftxt.write("\n".join(logs))

        print("Saved:", img_out)
        print("Saved:", log_out)

print("done")