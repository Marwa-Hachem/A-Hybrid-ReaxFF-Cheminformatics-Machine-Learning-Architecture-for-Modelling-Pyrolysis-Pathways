import os
import json
import re
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem import rdChemReactions

RDLogger.DisableLog("rdApp.*")

IN_CSV    = r"./.csv"
RXN_COL   = "rxn_smiles"
LABEL_COL = "true_label"
OUTDIR    = r"./"

RXN_MAPPED_COL = "rxn_smiles_mapped"

FP_RADIUS = 2
FP_NBITS  = 512

SKIP_SMALL_SPECIES = False
MIN_HEAVY_ATOMS    = 1
MIN_MOTIF_FREQ     = 5

CSV_ENCODINGS = ("utf-8", "cp1252", "latin1")

# IMPORTANT::
# For the ORIGINAL training/labeled set:
#   USE_EXISTING_VOCAB = False
# For unseen / VG / unlabeled / external data:
#   USE_EXISTING_VOCAB = True
#   and point EXISTING_VOCAB_JSON to the vocabulary built from training data

USE_EXISTING_VOCAB = True
EXISTING_VOCAB_JSON = r"./all_rxns_with_labels_features/motif_to_idx_CURRENT.json"

def read_csv_robust(path):
    last_err = None
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def split_reaction(rxn: str):
    rxn = (rxn or "").strip()
    if ">>" not in rxn:
        return [], []
    left, right = rxn.split(">>", 1)
    R = [m.strip() for m in left.split(".") if m.strip()]
    P = [m.strip() for m in right.split(".") if m.strip()]
    return R, P


def safe_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def is_too_small(mol) -> bool:
    if mol is None:
        return True
    if not SKIP_SMALL_SPECIES:
        return False
    return mol.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS


def brics_motifs_from_mol(mol):
    if mol is None:
        return set()
    try:
        return set(BRICS.BRICSDecompose(mol))
    except Exception:
        return set()


def motifs_from_side(smiles_list):
    motifs = set()
    for smi in smiles_list:
        mol = safe_mol(smi)
        if is_too_small(mol):
            continue
        motifs |= brics_motifs_from_mol(mol)
    return motifs


def build_motif_vocab_from_reactions(rxns):
    motif_counts = {}
    for rxn in rxns:
        R, P = split_reaction(rxn)
        if not R or not P:
            continue
        mR = motifs_from_side(R)
        mP = motifs_from_side(P)
        for m in (mR | mP):
            motif_counts[m] = motif_counts.get(m, 0) + 1

    motifs_kept = [m for m, c in motif_counts.items() if c >= MIN_MOTIF_FREQ]
    motifs_kept.sort()
    motif_to_idx = {m: i for i, m in enumerate(motifs_kept)}
    return motif_to_idx, motif_counts


def motif_vector_side(smiles_side, motif_to_idx: dict):
    V = len(motif_to_idx)
    v = np.zeros(V, dtype=np.uint8)
    motifs = motifs_from_side(smiles_side)
    for m in motifs:
        j = motif_to_idx.get(m)
        if j is not None:
            v[j] = 1
    return v


def motif_vectors_react_prod(rxn: str, motif_to_idx: dict):
    R, P = split_reaction(rxn)
    if not R or not P:
        return None, None, False, "bad_split"
    vR = motif_vector_side(R, motif_to_idx)
    vP = motif_vector_side(P, motif_to_idx)
    return vR, vP, True, ""


def morgan_bits(mol):
    if mol is None:
        return np.zeros(FP_NBITS, dtype=np.uint8)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    arr = np.zeros((FP_NBITS,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def agg_or_morgan(smiles_list):
    out = np.zeros(FP_NBITS, dtype=np.uint8)
    for smi in smiles_list:
        mol = safe_mol(smi)
        if mol is None:
            continue
        out |= morgan_bits(mol)
    return out


def count_rings(mol):
    try:
        return int(mol.GetRingInfo().NumRings()) if mol is not None else 0
    except Exception:
        return 0


def count_aromatic_atoms(mol):
    if mol is None:
        return 0
    try:
        return int(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()))
    except Exception:
        return 0


def count_aromatic_bonds(mol):
    if mol is None:
        return 0
    try:
        return int(sum(1 for b in mol.GetBonds() if b.GetIsAromatic()))
    except Exception:
        return 0


def count_carbonyls_relaxed(mol) -> int:
    if mol is None:
        return 0

    carbonyl_carbons = set()
    for c in mol.GetAtoms():
        if c.GetSymbol() != "C":
            continue

        for o in c.GetNeighbors():
            if o.GetSymbol() != "O":
                continue

            bond = mol.GetBondBetweenAtoms(c.GetIdx(), o.GetIdx())
            if bond is None:
                continue

            if bond.GetBondType() == Chem.BondType.DOUBLE:
                carbonyl_carbons.add(c.GetIdx())
                continue

            if c.GetHybridization() == Chem.HybridizationType.SP2 and o.GetDegree() == 1:
                carbonyl_carbons.add(c.GetIdx())
                continue

    return int(len(carbonyl_carbons))


def sum_over_side(smiles_list, fn):
    s = 0
    for smi in smiles_list:
        mol = safe_mol(smi)
        if mol is None:
            continue
        s += fn(mol)
    return int(s)


_BRACKET_TOKEN = re.compile(r"\[[^\]]+\]")


def count_brackets(smiles_list):
    return sum(len(_BRACKET_TOKEN.findall(smi)) for smi in smiles_list)


def count_bracket_carbons(smiles_list):
    pat = re.compile(r"\[(C|CH|CH2|CH3)\]")
    return sum(len(pat.findall(smi)) for smi in smiles_list)


def count_bracket_oxygens(smiles_list):
    pat = re.compile(r"\[(O|OH)\]")
    return sum(len(pat.findall(smi)) for smi in smiles_list)


def oxygen_edge_middle_counts(mol):
    if mol is None:
        return 0, 0, 0

    o_edge = 0
    o_middle = 0
    o_carbonyl = 0

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "O":
            continue

        deg = atom.GetDegree()

        is_carbonyl = False
        for b in atom.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE:
                other = b.GetOtherAtom(atom)
                if other.GetSymbol() == "C":
                    is_carbonyl = True
                    break

        if is_carbonyl:
            o_carbonyl += 1
            continue

        if deg == 1:
            o_edge += 1
        elif deg == 2:
            o_middle += 1

    return int(o_edge), int(o_middle), int(o_carbonyl)


def sum_oxygen_edge_middle(smiles_list):
    e = m = c = 0
    for smi in smiles_list:
        mol = safe_mol(smi)
        ee, mm, cc = oxygen_edge_middle_counts(mol)
        e += ee
        m += mm
        c += cc
    return int(e), int(m), int(c)


def radicals_count_mol(mol) -> int:
    if mol is None:
        return 0
    try:
        return int(sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
    except Exception:
        return 0


def radicals_count_smiles_fallback(smiles: str) -> int:
    if not isinstance(smiles, str):
        return 0
    pat = re.compile(r"\[(?:C|CH|CH2|CH3|O|OH|N|NH|NH2|S|SH)\]")
    return int(len(pat.findall(smiles)))


def sum_radicals_over_side(smiles_list):
    rdkit_sum = 0
    smi_proxy = 0
    for smi in smiles_list:
        mol = safe_mol(smi)
        rdkit_sum += radicals_count_mol(mol)
        smi_proxy += radicals_count_smiles_fallback(smi)
    return int(rdkit_sum), int(smi_proxy)


def has_radical_side(smiles_list):
    rdkit_sum, proxy_sum = sum_radicals_over_side(smiles_list)
    return int((rdkit_sum > 0) or (proxy_sum > 0))


def rxn_from_mapped_smiles(mapped_rxn: str):
    if not isinstance(mapped_rxn, str) or ">>" not in mapped_rxn:
        return None
    try:
        return rdChemReactions.ReactionFromSmarts(mapped_rxn, useSmiles=True)
    except Exception:
        return None


def _bond_set_and_ringinfo(mols):
    bonds = set()
    ring_bonds = set()
    aromatic_bonds = set()
    elem = {}
    atom_is_aromatic = {}

    for mol in mols:
        if mol is None:
            continue

        for a in mol.GetAtoms():
            m = a.GetAtomMapNum()
            if m:
                m = int(m)
                elem[m] = a.GetSymbol()
                atom_is_aromatic[m] = bool(a.GetIsAromatic())

        amap = [a.GetAtomMapNum() for a in mol.GetAtoms()]
        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            m1 = int(amap[i])
            m2 = int(amap[j])
            if m1 == 0 or m2 == 0:
                continue

            a, bmap = (m1, m2) if m1 < m2 else (m2, m1)
            bonds.add((a, bmap, str(b.GetBondType())))

            if b.IsInRing():
                ring_bonds.add((a, bmap))

            if b.GetIsAromatic():
                aromatic_bonds.add((a, bmap))

    return bonds, ring_bonds, aromatic_bonds, elem, atom_is_aromatic


def broken_formed_bonds_with_meta(rxn):
    reactants = list(rxn.GetReactants())
    products  = list(rxn.GetProducts())

    R_bonds, R_ring, R_arom_bonds, R_elem, R_atom_arom = _bond_set_and_ringinfo(reactants)
    P_bonds, P_ring, P_arom_bonds, P_elem, P_atom_arom = _bond_set_and_ringinfo(products)

    broken = []
    for (m1, m2, bt) in sorted(R_bonds - P_bonds):
        e1 = R_elem.get(m1, "?")
        e2 = R_elem.get(m2, "?")
        in_ring = (m1, m2) in R_ring
        is_aromatic_bond = (m1, m2) in R_arom_bonds
        broken.append((m1, m2, bt, e1, e2, in_ring, is_aromatic_bond))

    formed = []
    for (m1, m2, bt) in sorted(P_bonds - R_bonds):
        e1 = P_elem.get(m1, "?")
        e2 = P_elem.get(m2, "?")
        in_ring = (m1, m2) in P_ring
        is_aromatic_bond = (m1, m2) in P_arom_bonds
        formed.append((m1, m2, bt, e1, e2, in_ring, is_aromatic_bond))

    meta = {
        "R_atom_arom": R_atom_arom,
        "P_atom_arom": P_atom_arom,
    }

    return broken, formed, meta


def _merge_mols(mols):
    combo = None
    for mol in mols:
        if mol is None:
            continue
        combo = mol if combo is None else Chem.CombineMols(combo, mol)
    return combo

def _mapnum_to_atomidx(mol):
    d = {}
    if mol is None:
        return d
    for a in mol.GetAtoms():
        m = a.GetAtomMapNum()
        if m:
            d[int(m)] = a.GetIdx()
    return d


def carbonyl_carbons_mapnums_strict(mols):
    out = set()
    for mol in mols:
        if mol is None:
            continue
        for b in mol.GetBonds():
            if b.GetBondType() != Chem.BondType.DOUBLE:
                continue
            a1 = b.GetBeginAtom()
            a2 = b.GetEndAtom()
            s1, s2 = a1.GetSymbol(), a2.GetSymbol()
            if (s1, s2) == ("C", "O"):
                m = a1.GetAtomMapNum()
                if m:
                    out.add(int(m))
            elif (s1, s2) == ("O", "C"):
                m = a2.GetAtomMapNum()
                if m:
                    out.add(int(m))
    return out


def alpha_level_for_bond(rxn, m1, m2):
    reactants = list(rxn.GetReactants())
    carbonylCs = carbonyl_carbons_mapnums_strict(reactants)
    if not carbonylCs:
        return 99

    combo = _merge_mols(reactants)
    if combo is None:
        return 99

    map2idx = _mapnum_to_atomidx(combo)
    if m1 not in map2idx or m2 not in map2idx:
        return 99

    idx1 = map2idx[m1]
    idx2 = map2idx[m2]

    best = 99
    for c_map in carbonylCs:
        if c_map not in map2idx:
            continue
        c_idx = map2idx[c_map]

        if c_idx == idx1 or c_idx == idx2:
            return 0

        try:
            p1 = Chem.rdmolops.GetShortestPath(combo, c_idx, idx1)
            d1 = len(p1) - 1 if p1 else 99
        except Exception:
            d1 = 99

        try:
            p2 = Chem.rdmolops.GetShortestPath(combo, c_idx, idx2)
            d2 = len(p2) - 1 if p2 else 99
        except Exception:
            d2 = 99

        best = min(best, d1, d2)
        if best == 0:
            break

    return int(best if best != 99 else 99)


def _bond_type_bucket(bt: str) -> str:
    if bt == "SINGLE":
        return "single"
    if bt == "DOUBLE":
        return "double"
    if bt == "TRIPLE":
        return "triple"
    return "other"


def _pair_bucket(e1: str, e2: str) -> str:
    pair = "".join(sorted([e1, e2]))
    if pair == "CC":
        return "CC"
    if pair == "CO":
        return "CO"
    return "CX"

def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Missing: {IN_CSV}")

    os.makedirs(OUTDIR, exist_ok=True)

    df = read_csv_robust(IN_CSV)
    if RXN_COL not in df.columns:
        raise ValueError(f"Missing '{RXN_COL}'. Found: {list(df.columns)}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing '{LABEL_COL}'. Found: {list(df.columns)}")

    has_mapped = (RXN_MAPPED_COL in df.columns)
    if not has_mapped:
        print(f"[WARN] '{RXN_MAPPED_COL}' not found in CSV. Bond-change features depending on mapped reactions will stay at default values.")

    labels = (
        df[LABEL_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
    )
    has_label = (labels != "").astype(np.uint8)

    rxns = df[RXN_COL].astype(str).tolist()

    if USE_EXISTING_VOCAB:
        if not os.path.exists(EXISTING_VOCAB_JSON):
            raise FileNotFoundError(
                f"USE_EXISTING_VOCAB=True but vocabulary file not found:\n{EXISTING_VOCAB_JSON}"
            )

        with open(EXISTING_VOCAB_JSON, "r", encoding="utf-8") as f:
            motif_to_idx = json.load(f)

        motif_counts = {}
        V = len(motif_to_idx)

        print(f"Loaded existing motif vocabulary: V={V:,}")
        print(f"Vocabulary file: {EXISTING_VOCAB_JSON}")

        vocab_path = EXISTING_VOCAB_JSON
        counts_path = None

    else:
        motif_to_idx, motif_counts = build_motif_vocab_from_reactions(rxns)
        V = len(motif_to_idx)

        vocab_path = os.path.join(OUTDIR, "motif_to_idx_CURRENT.json")
        counts_path = os.path.join(OUTDIR, "motif_counts_CURRENT.json")

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(motif_to_idx, f, indent=2)

        with open(counts_path, "w", encoding="utf-8") as f:
            json.dump({k: int(v) for k, v in motif_counts.items()}, f, indent=2)

        print(f"Built CURRENT motif vocab: V={V:,} (MIN_MOTIF_FREQ={MIN_MOTIF_FREQ})")
        print(f"Saved vocabulary: {vocab_path}")
        print(f"Saved counts    : {counts_path}")

    mR_list, mP_list = [], []
    brics_R_list, brics_P_list, keep_idx = [], [], []
    brics_ok, brics_reason = [], []
    extras = []

    for i, rxn in enumerate(rxns):
        R, P = split_reaction(rxn)


        fR = agg_or_morgan(R)
        fP = agg_or_morgan(P)
        mR_list.append(fR)
        mP_list.append(fP)


        vR, vP, ok, reason = motif_vectors_react_prod(rxn, motif_to_idx)
        if ok:
            brics_R_list.append(vR)
            brics_P_list.append(vP)
            keep_idx.append(i)
            brics_ok.append(1)
            brics_reason.append("")
        else:
            brics_ok.append(0)
            brics_reason.append(reason)

        R_O_edge, R_O_middle, R_O_carbonyl = sum_oxygen_edge_middle(R)
        P_O_edge, P_O_middle, P_O_carbonyl = sum_oxygen_edge_middle(P)

        R_rad_rdkit, R_rad_proxy = sum_radicals_over_side(R)
        P_rad_rdkit, P_rad_proxy = sum_radicals_over_side(P)

        R_has_radical = has_radical_side(R)
        P_has_radical = has_radical_side(P)

        n_broken = 0
        n_formed = 0
        n_broken_in_ring = 0
        n_formed_in_ring = 0
        n_aromatic_broken = 0
        n_aromatic_formed = 0

        broken_CC = broken_CO = broken_CX = 0
        formed_CC = formed_CO = formed_CX = 0

        broken_single = broken_double = broken_triple = broken_other = 0
        formed_single = formed_double = formed_triple = formed_other = 0

        broken_alpha_cnt = broken_beta_cnt = broken_gamma_cnt = broken_far_cnt = 0
        broken_min_alpha_level = 99

        n_reactive_atoms_reactants = 0
        n_reactive_atoms_products = 0
        n_reactive_aromatic_atoms_reactants = 0
        n_reactive_aromatic_atoms_products = 0

        if has_mapped:
            mapped_rxn = df.loc[i, RXN_MAPPED_COL]
            rxn_obj = rxn_from_mapped_smiles(mapped_rxn)

            if rxn_obj is not None:
                broken, formed, meta = broken_formed_bonds_with_meta(rxn_obj)

                n_broken = len(broken)
                n_formed = len(formed)
                n_broken_in_ring = sum(1 for x in broken if x[5])
                n_formed_in_ring = sum(1 for x in formed if x[5])
                n_aromatic_broken = sum(1 for x in broken if x[6])
                n_aromatic_formed = sum(1 for x in formed if x[6])

                reactive_atoms_R = set()
                reactive_atoms_P = set()

                for (m1, m2, bt, e1, e2, in_ring, is_aromatic_bond) in broken:
                    reactive_atoms_R.add(m1)
                    reactive_atoms_R.add(m2)

                    pb = _pair_bucket(e1, e2)
                    if pb == "CC":
                        broken_CC += 1
                    elif pb == "CO":
                        broken_CO += 1
                    else:
                        broken_CX += 1

                    bb = _bond_type_bucket(bt)
                    if bb == "single":
                        broken_single += 1
                    elif bb == "double":
                        broken_double += 1
                    elif bb == "triple":
                        broken_triple += 1
                    else:
                        broken_other += 1

                    lvl = alpha_level_for_bond(rxn_obj, m1, m2)
                    broken_min_alpha_level = min(broken_min_alpha_level, lvl)

                    if lvl == 0:
                        broken_alpha_cnt += 1
                    elif lvl == 1:
                        broken_beta_cnt += 1
                    elif lvl == 2:
                        broken_gamma_cnt += 1
                    elif lvl >= 3 and lvl != 99:
                        broken_far_cnt += 1

                for (m1, m2, bt, e1, e2, in_ring, is_aromatic_bond) in formed:
                    reactive_atoms_P.add(m1)
                    reactive_atoms_P.add(m2)

                    pb = _pair_bucket(e1, e2)
                    if pb == "CC":
                        formed_CC += 1
                    elif pb == "CO":
                        formed_CO += 1
                    else:
                        formed_CX += 1

                    bb = _bond_type_bucket(bt)
                    if bb == "single":
                        formed_single += 1
                    elif bb == "double":
                        formed_double += 1
                    elif bb == "triple":
                        formed_triple += 1
                    else:
                        formed_other += 1

                n_reactive_atoms_reactants = len(reactive_atoms_R)
                n_reactive_atoms_products = len(reactive_atoms_P)

                R_atom_arom = meta.get("R_atom_arom", {})
                P_atom_arom = meta.get("P_atom_arom", {})

                n_reactive_aromatic_atoms_reactants = sum(
                    1 for m in reactive_atoms_R if R_atom_arom.get(m, False)
                )
                n_reactive_aromatic_atoms_products = sum(
                    1 for m in reactive_atoms_P if P_atom_arom.get(m, False)
                )

        feats = {
            "n_reactants": len(R),
            "n_products": len(P),
            "d_species": len(P) - len(R),
            "parse_ok": int(bool(R) and bool(P)),

            "R_rings": sum_over_side(R, count_rings),
            "P_rings": sum_over_side(P, count_rings),
            "R_carbonyls": sum_over_side(R, count_carbonyls_relaxed),
            "P_carbonyls": sum_over_side(P, count_carbonyls_relaxed),

            "R_aromatic_atoms": sum_over_side(R, count_aromatic_atoms),
            "P_aromatic_atoms": sum_over_side(P, count_aromatic_atoms),
            "R_aromatic_bonds": sum_over_side(R, count_aromatic_bonds),
            "P_aromatic_bonds": sum_over_side(P, count_aromatic_bonds),

            "R_radicals": count_brackets(R),
            "P_radicals": count_brackets(P),
            "R_radical_C": count_bracket_carbons(R),
            "P_radical_C": count_bracket_carbons(P),
            "R_radical_O": count_bracket_oxygens(R),
            "P_radical_O": count_bracket_oxygens(P),
            "radical_change": count_brackets(R) - count_brackets(P),

            "R_O_edge": R_O_edge,
            "P_O_edge": P_O_edge,
            "R_O_middle": R_O_middle,
            "P_O_middle": P_O_middle,
            "R_O_carbonyl": R_O_carbonyl,
            "P_O_carbonyl": P_O_carbonyl,

            # "R_radicals_rdkit": R_rad_rdkit,
            # "P_radicals_rdkit": P_rad_rdkit,
            # "R_radicals_proxy": R_rad_proxy,
            # "P_radicals_proxy": P_rad_proxy,

            "R_has_radical": R_has_radical,
            "P_has_radical": P_has_radical,
            "radical_change_general": P_has_radical - R_has_radical,

            "n_bonds_broken": n_broken,
            "n_bonds_formed": n_formed,
            "n_broken_in_ring": n_broken_in_ring,
            "n_formed_in_ring": n_formed_in_ring,
            "n_aromatic_broken": n_aromatic_broken,
            "n_aromatic_formed": n_aromatic_formed,

            "n_reactive_atoms_reactants": n_reactive_atoms_reactants,
            "n_reactive_atoms_products": n_reactive_atoms_products,
            "n_reactive_aromatic_atoms_reactants": n_reactive_aromatic_atoms_reactants,
            "n_reactive_aromatic_atoms_products": n_reactive_aromatic_atoms_products,

            "broken_CC": broken_CC,
            "broken_CO": broken_CO,
            "broken_CX": broken_CX,

            "formed_CC": formed_CC,
            "formed_CO": formed_CO,
            "formed_CX": formed_CX,

            "broken_single": broken_single,
            "broken_double": broken_double,
            "broken_triple": broken_triple,
            "broken_other": broken_other,

            "formed_single": formed_single,
            "formed_double": formed_double,
            "formed_triple": formed_triple,
            "formed_other": formed_other,

            # "broken_alpha_cnt": broken_alpha_cnt,
            # "broken_beta_cnt": broken_beta_cnt,
            # "broken_gamma_cnt": broken_gamma_cnt,
            # "broken_far_cnt": broken_far_cnt,
            # "broken_min_alpha_level": int(broken_min_alpha_level) if broken_min_alpha_level != 99 else 99,
        }

        extras.append(feats)

    morgan_R = np.stack(mR_list).astype(np.uint8)
    morgan_P = np.stack(mP_list).astype(np.uint8)

    keep_idx_arr = np.array(keep_idx, dtype=np.int64)

    brics_react = np.stack(brics_R_list).astype(np.uint8) if brics_R_list else np.zeros((0, V), dtype=np.uint8)
    brics_prod  = np.stack(brics_P_list).astype(np.uint8) if brics_P_list else np.zeros((0, V), dtype=np.uint8)

    extra_df = pd.DataFrame(extras)

    out_df = pd.concat([df.reset_index(drop=True), extra_df], axis=1)
    out_df["has_label"] = has_label
    out_df["has_brics_fp"] = brics_ok
    out_df["brics_fp_fail_reason"] = brics_reason

    out_csv = os.path.join(OUTDIR, "rxns_with_features_CURRENT.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    out_npz = os.path.join(OUTDIR, "features_arrays_CURRENT.npz")
    np.savez_compressed(
        out_npz,
        rxn_smiles=np.array(rxns, dtype=object),

        labels=np.array(labels.tolist(), dtype=object),
        has_label=np.array(has_label.tolist(), dtype=np.uint8),

        morgan_reactant=morgan_R,
        morgan_product=morgan_P,

        brics_reactant=brics_react,
        brics_product=brics_prod,
        brics_keep_idx=keep_idx_arr,

        extras=extra_df.to_numpy(dtype=np.float32),
        extra_feature_names=np.array(extra_df.columns.tolist(), dtype=object),

        motif_vocab_size=np.array([V], dtype=np.int64),
        min_motif_freq=np.array([MIN_MOTIF_FREQ], dtype=np.int64),
    )

    n = len(rxns)
    n_lab = int(has_label.sum())
    n_parse_ok = int(extra_df["parse_ok"].sum())
    n_brics_ok = int(np.sum(brics_ok))

    print("\n[DONE]")
    print("Saved CSV :", out_csv)
    print("Saved NPZ :", out_npz)
    print("\nSHAPES")
    print("N =", n)
    print("Manual labeled =", n_lab)
    print("Parse OK =", n_parse_ok)
    print("Morgan R/P =", morgan_R.shape, morgan_P.shape)
    print("BRICS react/prod =", brics_react.shape, brics_prod.shape, "(aligned by keep_idx len =", len(keep_idx_arr), ")")
    print("Extras =", extra_df.shape)
    print(f"BRICS ok rows = {n_brics_ok}/{n} ({(100*n_brics_ok/max(n,1)):.1f}%)")

    if USE_EXISTING_VOCAB:
        print("\nMode: reused existing BRICS vocabulary")
        print(f"Vocabulary source: {EXISTING_VOCAB_JSON}")
    else:
        print("\nMode: built new BRICS vocabulary from this dataset")
        print(f"Vocabulary saved at: {vocab_path}")


if __name__ == "__main__":
    main()