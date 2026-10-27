import glob
import re
import os
from collections import Counter, defaultdict

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw, rdChemReactions
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")


MIN_COUNT = 1
log_files = glob.glob("./CHECK-Annealing- pathway_logs/*_path_*.txt")

out_dir = "./CHECK-Annealing- pathway_outputs"
img_dir = os.path.join(out_dir, "images")
log_dir = os.path.join(out_dir, "logs")
plot_dir = os.path.join(out_dir, "plots")

os.makedirs(img_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


def extract_m_blocks(text: str):
    """
    Extract SMILES from M#(...) or M# (...) blocks, preserving nested parentheses.

    Example:
        'M4 ([CH2]CCC[C]([CH2])[O]), M9 ([OH])'
    returns:
        ['[CH2]CCC[C]([CH2])[O]', '[OH]']
    """
    out = []
    i = 0
    n = len(text)

    while i < n:
        m = re.search(r"M\d+\s*\(", text[i:])
        if not m:
            break

        start = i + m.start()
        open_paren = i + m.end() - 1
        j = open_paren + 1
        depth = 1

        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1

        if depth == 0:
            smiles = text[open_paren + 1 : j - 1].strip()
            out.append(smiles)
            i = j
        else:
            break

    return out


def split_step_line(line: str):
    """
    Parse a log line like:
      Step 1: M1 (...), M2 (...) → M3 (...), M4 (...)

    Returns:
      (reactants_list, products_list)

    If parsing fails, returns ([], []).
    """
    line = line.strip()

    if not line.startswith("Step"):
        return [], []

    if ":" not in line:
        return [], []

    rhs = line.split(":", 1)[1].strip()

    if "→" in rhs:
        left, right = rhs.split("→", 1)
    elif "->" in rhs:
        left, right = rhs.split("->", 1)
    else:
        return [], []

    reactants = extract_m_blocks(left)
    products = extract_m_blocks(right)

    return reactants, products


def canonicalize_step(reactants, products):
    return (
        tuple(sorted(r.strip() for r in reactants if r.strip())),
        tuple(sorted(p.strip() for p in products if p.strip())),
    )


def step_to_string(step):
    reactants, products = step
    return f"{', '.join(reactants)} -> {', '.join(products)}"


def extract_pathway(file_path):
    steps = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            reactants, products = split_step_line(line.strip())
            if reactants and products:
                steps.append(canonicalize_step(reactants, products))

    return tuple(steps)


def draw_step(reactant_smis, product_smis, step_idx):
    """
    Draw one full reaction step with multiple reactants/products.
    """
    rxn = rdChemReactions.ChemicalReaction()

    for smi in reactant_smis:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                rxn.AddReactantTemplate(mol)
        except Exception:
            pass

    for smi in product_smis:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                rxn.AddProductTemplate(mol)
        except Exception:
            pass

    core = Draw.ReactionToImage(rxn, subImgSize=(300, 200))

    font = ImageFont.load_default()
    caption = f"Step {step_idx}: {', '.join(reactant_smis)} → {', '.join(product_smis)}"

    img = Image.new("RGB", (core.size[0], core.size[1] + 45), "white")
    img.paste(core, (0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((5, core.size[1] + 5), caption, fill=(0, 0, 0), font=font)
    return img


def add_header(images, header_text):
    if not images:
        return None

    font = ImageFont.load_default()
    w = max(im.size[0] for im in images)
    h = sum(im.size[1] for im in images)

    header_h = font.getbbox("Ag")[3] + 22
    total_h = h + header_h

    out = Image.new("RGB", (w, total_h), "white")
    draw = ImageDraw.Draw(out)
    draw.text((10, 5), header_text, fill=(0, 0, 0), font=font)

    y = header_h
    for im in images:
        out.paste(im, (0, y))
        y += im.size[1]

    return out


all_paths = []
source_files_by_path = defaultdict(list)

print(f"Found {len(log_files)} log files.")

for log in log_files:
    pw = extract_pathway(log)

    if pw:
        all_paths.append(pw)
        source_files_by_path[pw].append(log)
    else:
        print(f"Skipped (no parsed steps): {log}")

counts = Counter(all_paths)

filtered_counts = Counter({p: f for p, f in counts.items() if f > MIN_COUNT})
path_items = filtered_counts.most_common()
total = sum(filtered_counts.values())

print("\nPathway Statistics")
print(f"Filter: keep pathways with frequency > {MIN_COUNT} (i.e., >= {MIN_COUNT + 1})")
print(f"Total pathways kept: {total}")
print(f"Unique pathways kept: {len(filtered_counts)}\n")

if not path_items:
    print(f"No pathways appear more than {MIN_COUNT} times. Nothing to save.")
    raise SystemExit(0)

percents = [100.0 * freq / total for _, freq in path_items] if total > 0 else []
if percents:
    percents[-1] += (100.0 - sum(percents))


summary_path = os.path.join(log_dir, "pathway_summary.txt")
with open(summary_path, "w", encoding="utf-8") as sf:
    sf.write("Pathway Summary\n")
    sf.write(f"Filter: keep pathways with frequency > {MIN_COUNT} (i.e., >= {MIN_COUNT + 1})\n")
    sf.write(f"Total pathways kept: {total}\n")
    sf.write(f"Unique pathways kept: {len(filtered_counts)}\n\n")

    for i, ((path, freq), perc) in enumerate(zip(path_items, percents), 1):
        sf.write(f"Pathway {i}: Appears {freq} times ({perc:.2f}%)\n")

        for step_idx, step in enumerate(path, 1):
            sf.write(f"   Step {step_idx}: {step_to_string(step)}\n")

        sf.write(" - Source files:\n")
        for fp in source_files_by_path[path][:20]:
            sf.write(f" {fp}\n")
        if len(source_files_by_path[path]) > 20:
            sf.write(f" (+{len(source_files_by_path[path]) - 20} more)\n")

        sf.write("\n")

print(f"Saved summary log: {summary_path}\n")


for i, ((path, freq), perc) in enumerate(zip(path_items, percents), 1):
    per_path_log = os.path.join(log_dir, f"pathway_{i:03d}.txt")
    with open(per_path_log, "w", encoding="utf-8") as pf:
        pf.write(f"Pathway {i}\n")
        pf.write(f"Frequency: {freq} ({perc:.2f}%)\n\n")

        for step_idx, step in enumerate(path, 1):
            pf.write(f"Step {step_idx}: {step_to_string(step)}\n")

        pf.write("\nSource files:\n")
        for fp in source_files_by_path[path]:
            pf.write(f"- {fp}\n")

    step_imgs = []
    for step_idx, step in enumerate(path, 1):
        reactants, products = step
        step_imgs.append(draw_step(list(reactants), list(products), step_idx))

    header_text = f"Pathway {i} — Appears {freq} times ({perc:.2f}%)"
    final_img = add_header(step_imgs, header_text)

    out_img = os.path.join(img_dir, f"CHECK-pathway_{i:03d}_FULL.png")
    if final_img is not None:
        final_img.save(out_img)
        print(f"Saved image: {out_img}")

    print(f"Saved log:   {per_path_log}\n")
