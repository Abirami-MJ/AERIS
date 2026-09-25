# ============================================================
# AERIS Project — Dempster-Shafer Evidence Fusion
# ============================================================

import json
import glob

import os
os.makedirs("DempsterShafer", exist_ok=True)

# ------------------------------------------------------------
# 1. LOAD ALL FUSION EVIDENCE FILES
# ------------------------------------------------------------
cmapss_files = glob.glob("Fusion_Evidence/CMAPSS/cmapss_fusion_evidence_97.json")
ntsb_file = "Fusion_Evidence/NTSB/ntsb_fusion_evidence.json"

with open(ntsb_file, "r") as f:
    ntsb_evidence_list = json.load(f)

print(f"CMAPSS fusion evidence files: {len(cmapss_files)}")
print(f"NTSB fusion evidence records: {len(ntsb_evidence_list)}")

# ------------------------------------------------------------
# 2. DEMPSTER-SHAFER COMBINATION RULE (for two sources, single hypothesis + uncertainty)
# ------------------------------------------------------------
def combine_two(evidence_a, evidence_b):
    """
    Combines two pieces of evidence using Dempster's rule of combination,
    simplified for the case of ONE hypothesis + uncertainty (no explicit
    conflicting hypothesis mass — a common simplification when each source
    only expresses belief in "risk level X" vs "unknown").
    """
    m1_belief, m1_unc = evidence_a["belief"], evidence_a["uncertainty"]
    m2_belief, m2_unc = evidence_b["belief"], evidence_b["uncertainty"]

    same_hypothesis = evidence_a["hypothesis"] == evidence_b["hypothesis"]

    if same_hypothesis:
        # Both sources agree on the hypothesis -> combine belief masses
        combined_belief = m1_belief * m2_belief + m1_belief * m2_unc + m1_unc * m2_belief
        combined_uncertainty = m1_unc * m2_unc
        conflict = 0.0
        fused_hypothesis = evidence_a["hypothesis"]
    else:
        # Sources disagree -> there is genuine conflict (K), which DS theory
        # normalizes out. This is a simplified two-hypothesis conflict case.
        conflict = m1_belief * m2_belief
        normalization = 1 - conflict
        if normalization <= 0:
            # Total conflict -> cannot combine meaningfully
            return {
                "hypothesis": "Conflicting",
                "belief": 0.0,
                "uncertainty": 1.0,
                "conflict": conflict
            }
        # Whichever source has higher belief "wins" the fused hypothesis,
        # weighted by normalization -- this is a simplification worth
        # refining if your project needs full multi-hypothesis DS theory.
        if m1_belief >= m2_belief:
            fused_hypothesis = evidence_a["hypothesis"]
            combined_belief = (m1_belief * m2_unc + m1_unc * m2_belief * 0) / normalization
        else:
            fused_hypothesis = evidence_b["hypothesis"]
            combined_belief = (m2_belief * m1_unc) / normalization
        combined_uncertainty = (m1_unc * m2_unc) / normalization

    return {
        "hypothesis": fused_hypothesis,
        "belief": round(combined_belief, 3),
        "uncertainty": round(combined_uncertainty, 3),
        "conflict": round(conflict, 3)
    }

# ------------------------------------------------------------
# 3. EXAMPLE: FUSE ONE CMAPSS RECORD WITH ONE NTSB RECORD
# ------------------------------------------------------------
# NOTE: This is where the earlier "different real-world entities" problem
# applies. This example fuses the FIRST record from each source purely to
# demonstrate the mechanism -- not because they represent the same aircraft.

with open(cmapss_files[0], "r") as f:
    cmapss_example = json.load(f)

ntsb_example = ntsb_evidence_list[0]

print("\nCMAPSS Evidence:", json.dumps(cmapss_example, indent=2))
print("\nNTSB Evidence:", json.dumps(ntsb_example, indent=2))

fused_result = combine_two(cmapss_example, ntsb_example)

print("\n" + "=" * 50)
print("UNIFIED FUSED EVIDENCE (Dempster-Shafer)")
print("=" * 50)
print(json.dumps(fused_result, indent=2))

with open("DempsterShafer/Unified_Fused_Evidence.json", "w") as f:
    json.dump(fused_result, f, indent=2)