# ============================================================
# AERIS Project — Dempster-Shafer Evidence Fusion (Final Stage)
# Combines CMAPSS, NTSB, and NGAFID fusion evidence into one
# unified aircraft risk assessment using proper multi-hypothesis
# Dempster-Shafer combination (not a two-hypothesis shortcut).
# ============================================================

import json
import glob
import os
from itertools import product

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------
CMAPSS_FUSION_DIR = "Fusion_Evidence/CMAPSS"
NTSB_FUSION_DIR = "Fusion_Evidence/NTSB"
NGAFID_FUSION_DIR = "Fusion_Evidence/NGAFID"
OUTPUT_DIR = "DempsterShafer"

# Frame of discernment: the complete set of mutually exclusive
# hypotheses this system reasons over.
FRAME = frozenset({"Low Risk", "Medium Risk", "High Risk"})


# ------------------------------------------------------------
# 1. LOAD ONE FUSION EVIDENCE FILE PER SOURCE
# ------------------------------------------------------------
def load_single_fusion_file(folder, source_name):
    files = glob.glob(f"{folder}/*.json")
    if len(files) == 0:
        raise FileNotFoundError(f"No fusion evidence file found in {folder}")
    if len(files) > 1:
        print(f"WARNING: {len(files)} files found in {folder}; using the first: {files[0]}")
    with open(files[0], "r") as f:
        data = json.load(f)
    print(f"Loaded {source_name}: hypothesis='{data['hypothesis']}', belief={data['belief']}, uncertainty={data['uncertainty']}")
    return data

cmapss_ev = load_single_fusion_file(CMAPSS_FUSION_DIR, "CMAPSS")
ntsb_ev = load_single_fusion_file(NTSB_FUSION_DIR, "NTSB")
ngafid_ev = load_single_fusion_file(NGAFID_FUSION_DIR, "NGAFID")


# ------------------------------------------------------------
# 2. CONVERT EACH SOURCE INTO A PROPER MASS FUNCTION
# ------------------------------------------------------------
# Each source is modeled as a "simple support function": it assigns
# belief mass to ONE specific hypothesis (singleton), and the remaining
# mass to the full frame Theta (representing "don't know" / ignorance).
# This is a standard, textbook DS construction -- not a simplification
# unique to this project.

def build_mass_function(hypothesis, belief, uncertainty):
    singleton = frozenset({hypothesis})
    mass = {
        singleton: belief,
        FRAME: uncertainty
    }
    return mass

m_cmapss = build_mass_function(cmapss_ev["hypothesis"], cmapss_ev["belief"], cmapss_ev["uncertainty"])
m_ntsb = build_mass_function(ntsb_ev["hypothesis"], ntsb_ev["belief"], ntsb_ev["uncertainty"])
m_ngafid = build_mass_function(ngafid_ev["hypothesis"], ngafid_ev["belief"], ngafid_ev["uncertainty"])


# ------------------------------------------------------------
# 3. DEMPSTER'S RULE OF COMBINATION (general, not a two-case shortcut)
# ------------------------------------------------------------
def combine_mass_functions(m1, m2):
    """
    Implements Dempster's rule of combination in full generality:
        m12(C) = (1 / (1-K)) * sum over all A,B where A∩B=C of m1(A)*m2(B)
    where K (the conflict mass) = sum over all A,B where A∩B=empty of m1(A)*m2(B)

    Returns: (combined_mass_dict, conflict_K)
    """
    combined = {}
    conflict = 0.0

    for (A, mA), (B, mB) in product(m1.items(), m2.items()):
        intersection = A & B
        product_mass = mA * mB

        if len(intersection) == 0:
            conflict += product_mass
        else:
            combined[intersection] = combined.get(intersection, 0.0) + product_mass

    normalization = 1 - conflict
    if normalization <= 0:
        raise ValueError(
            f"Total conflict (K={conflict:.4f}) -- sources are completely "
            f"contradictory. Cannot combine meaningfully."
        )

    combined_normalized = {k: round(v / normalization, 6) for k, v in combined.items()}
    return combined_normalized, round(conflict, 6)


# ------------------------------------------------------------
# 4. SEQUENTIALLY COMBINE ALL THREE SOURCES
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("DEMPSTER-SHAFER SEQUENTIAL COMBINATION")
print("=" * 70)

print("\nStep 1: Combining CMAPSS + NTSB")
m_step1, conflict_1 = combine_mass_functions(m_cmapss, m_ntsb)
print(f"  Conflict (K): {conflict_1}")
for hyp, mass in m_step1.items():
    label = "Θ (ignorance)" if hyp == FRAME else list(hyp)[0]
    print(f"  m({label}) = {mass}")

print("\nStep 2: Combining (CMAPSS+NTSB) + NGAFID")
m_final, conflict_2 = combine_mass_functions(m_step1, m_ngafid)
print(f"  Conflict (K): {conflict_2}")
for hyp, mass in m_final.items():
    label = "Θ (ignorance)" if hyp == FRAME else list(hyp)[0]
    print(f"  m({label}) = {mass}")

if conflict_2 > 0.3:
    print(f"\n  NOTE: High conflict ({conflict_2}) detected at this step.")
    print(f"  This means NGAFID's evidence substantively disagreed with the")
    print(f"  already-combined CMAPSS+NTSB belief. This is EXPECTED and worth")
    print(f"  discussing in your report: these three sources describe different")
    print(f"  underlying aircraft/engines (a known limitation of using independent")
    print(f"  benchmark datasets), so disagreement here reflects that, not a bug.")


# ------------------------------------------------------------
# 5. DERIVED DS MEASURES: BELIEF, PLAUSIBILITY, PIGNISTIC PROBABILITY
# ------------------------------------------------------------
def belief_measure(hypothesis_singleton, mass_function):
    """Bel(A) = sum of masses of all subsets of A (only itself, for a singleton)."""
    return mass_function.get(frozenset({hypothesis_singleton}), 0.0)

def plausibility_measure(hypothesis_singleton, mass_function):
    """Pl(A) = sum of masses of all sets that intersect A."""
    target = frozenset({hypothesis_singleton})
    return sum(mass for focal_set, mass in mass_function.items() if focal_set & target)

def pignistic_probability(hypothesis_singleton, mass_function, frame):
    """
    Smets' pignistic transformation: converts a DS mass function into a
    single probability distribution for decision-making, by splitting
    ignorance mass equally across all hypotheses it could refer to.
    BetP(A) = sum over focal elements B containing A of m(B)/|B|
    """
    target = frozenset({hypothesis_singleton})
    total = 0.0
    for focal_set, mass in mass_function.items():
        if target.issubset(focal_set):
            total += mass / len(focal_set)
    return total

print("\n" + "=" * 70)
print("FINAL DERIVED RISK ASSESSMENT")
print("=" * 70)

results = {}
for hypothesis in FRAME:
    bel = belief_measure(hypothesis, m_final)
    pl = plausibility_measure(hypothesis, m_final)
    pignistic = pignistic_probability(hypothesis, m_final, FRAME)
    results[hypothesis] = {
        "belief": round(bel, 4),
        "plausibility": round(pl, 4),
        "pignistic_probability": round(pignistic, 4)
    }
    print(f"\n{hypothesis}:")
    print(f"  Belief (lower bound)        : {bel:.4f}")
    print(f"  Plausibility (upper bound)  : {pl:.4f}")
    print(f"  Pignistic probability       : {pignistic:.4f}")

final_decision = max(results.items(), key=lambda x: x[1]["pignistic_probability"])
print(f"\n>>> FINAL UNIFIED RISK DECISION: {final_decision[0]} "
      f"(pignistic probability: {final_decision[1]['pignistic_probability']:.4f})")


# ------------------------------------------------------------
# 6. SAVE UNIFIED RESULT
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

unified_output = {
    "sources_combined": ["CMAPSS", "NTSB", "NGAFID"],
    "source_summary": {
        "CMAPSS": {"hypothesis": cmapss_ev["hypothesis"], "belief": cmapss_ev["belief"]},
        "NTSB": {"hypothesis": ntsb_ev["hypothesis"], "belief": ntsb_ev["belief"]},
        "NGAFID": {"hypothesis": ngafid_ev["hypothesis"], "belief": ngafid_ev["belief"]},
    },
    "pairwise_conflicts": {
        "cmapss_ntsb_conflict": conflict_1,
        "combined_ngafid_conflict": conflict_2
    },
    "final_mass_distribution": {
        ("Θ (ignorance)" if k == FRAME else list(k)[0]): v for k, v in m_final.items()
    },
    "hypothesis_assessment": {h: results[h] for h in FRAME},
    "final_decision": {
        "risk_level": final_decision[0],
        "pignistic_probability": final_decision[1]["pignistic_probability"]
    }
}

output_path = os.path.join(OUTPUT_DIR, "Unified_Aircraft_Risk_Assessment.json")
with open(output_path, "w") as f:
    json.dump(unified_output, f, indent=2)

print(f"\nSaved unified risk assessment to: {output_path}")