# ============================================================
# AERIS Project — NTSB Fusion Evidence Generator (Research-Grade)
# Converts the Investigation Evidence Object into a Dempster-Shafer
# ready belief/uncertainty representation, with cross-source
# consistency checking against CMAPSS.
# ============================================================

import json
import glob
import os

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
NTSB_EVIDENCE_DIR = "evidence_objects/NTSB"
CMAPSS_EVIDENCE_DIR = "evidence_objects"   # your existing CMAPSS evidence folder
OUTPUT_DIR = "Fusion_Evidence/NTSB"

# Bonus applied to belief when CMAPSS and NTSB independently agree on
# the same subsystem -- capped so it can never push belief above 1.0.
CROSS_SOURCE_AGREEMENT_BONUS = 0.08

# ------------------------------------------------------------
# 1. RISK HYPOTHESIS DETERMINATION (explainable, rule-based)
# ------------------------------------------------------------
def determine_hypothesis(evidence):
    """
    Determines the risk hypothesis this historical case implies,
    based on real severity indicators from the investigation record --
    not the retrieval similarity, which only measures topical relevance.
    """
    causal = evidence["causal_evidence"]
    damage = evidence["investigation_summary"].get("damage", "Unknown").lower()

    if causal.get("fatal_injuries", 0) > 0:
        return "High Risk", "Historical case involved fatal injuries"
    if causal.get("serious_injuries", 0) > 0:
        return "High Risk", "Historical case involved serious injuries"
    if damage == "destroyed":
        return "High Risk", "Historical case resulted in aircraft destruction"
    if damage == "substantial":
        return "Medium Risk", "Historical case resulted in substantial aircraft damage"
    if damage == "minor":
        return "Low Risk", "Historical case resulted in only minor damage"

    # Damage unknown -- fall back on causal narrative severity as a proxy
    findings_text = causal.get("findings", "").lower()
    if "failure" in findings_text and "incorrect service" in findings_text:
        return "Medium Risk", "Failure with a maintenance-related contributing factor, but damage severity unrecorded"
    return "Medium Risk", "Damage severity unrecorded; defaulting to Medium Risk based on failure occurrence"


# ------------------------------------------------------------
# 2. CROSS-SOURCE CONSISTENCY CHECK
# ------------------------------------------------------------
def find_matching_cmapss_evidence(ntsb_evidence):
    """
    Attempts to locate the CMAPSS evidence object that generated this
    NTSB retrieval query, so we can check subsystem agreement between
    two independently-derived sources.
    """
    query = ntsb_evidence["retrieval_assessment"]["semantic_query"]

    for filepath in glob.glob(f"{CMAPSS_EVIDENCE_DIR}/cmapss_evidence_*.json"):
        with open(filepath, "r") as f:
            cmapss_ev = json.load(f)
        if cmapss_ev.get("faiss_query") == query:
            return cmapss_ev
    return None


def check_subsystem_agreement(ntsb_evidence, cmapss_evidence):
    if cmapss_evidence is None:
        return False, None, None

    ntsb_subsystem = ntsb_evidence["engineering_guidance"]["recommended_subsystem"]
    cmapss_subsystem = cmapss_evidence.get("probable_cause_subsystem")

    agrees = (ntsb_subsystem == cmapss_subsystem)
    return agrees, ntsb_subsystem, cmapss_subsystem


# ------------------------------------------------------------
# 3. MAIN CONVERSION
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

ntsb_files = glob.glob(f"{NTSB_EVIDENCE_DIR}/ntsb_evidence_*.json")
print(f"Found {len(ntsb_files)} NTSB investigation evidence object(s) to convert.")

for filepath in ntsb_files:
    with open(filepath, "r") as f:
        ntsb_evidence = json.load(f)

    base_confidence = ntsb_evidence["evidence_quality"]["overall_evidence_confidence"]
    hypothesis, hypothesis_reason = determine_hypothesis(ntsb_evidence)

    # ---- Cross-source consistency check ----
    cmapss_match = find_matching_cmapss_evidence(ntsb_evidence)
    agrees, ntsb_sub, cmapss_sub = check_subsystem_agreement(ntsb_evidence, cmapss_match)

    final_belief = base_confidence
    consistency_note = "No matching CMAPSS evidence found for cross-check."
    if cmapss_match is not None:
        if agrees:
            final_belief = min(base_confidence + CROSS_SOURCE_AGREEMENT_BONUS, 1.0)
            consistency_note = (
                f"CMAPSS and NTSB independently agree on subsystem "
                f"('{cmapss_sub}'). Belief boosted by {CROSS_SOURCE_AGREEMENT_BONUS}."
            )
        else:
            consistency_note = (
                f"CMAPSS predicted '{cmapss_sub}' but NTSB retrieval points to "
                f"'{ntsb_sub}'. No agreement bonus applied; belief unchanged."
            )

    final_belief = round(final_belief, 3)
    uncertainty = round(1 - final_belief, 3)

    fusion_evidence = {
        "source": "NTSB",
        "event_id": ntsb_evidence["metadata"]["event_id"],
        "hypothesis": hypothesis,
        "hypothesis_reason": hypothesis_reason,
        "belief": final_belief,
        "uncertainty": uncertainty,
        "cross_source_consistency": {
            "checked_against": "CMAPSS",
            "subsystems_agree": agrees,
            "ntsb_subsystem": ntsb_sub,
            "cmapss_subsystem": cmapss_sub,
            "note": consistency_note
        },
        "base_evidence_confidence": base_confidence,
        "recommended_subsystem": ntsb_evidence["engineering_guidance"]["recommended_subsystem"],
        "supporting_report": ntsb_evidence["engineering_guidance"]["supporting_report"]
    }

    output_path = os.path.join(OUTPUT_DIR, f"ntsb_fusion_{ntsb_evidence['metadata']['event_id']}.json")
    with open(output_path, "w") as f:
        json.dump(fusion_evidence, f, indent=2)

    print(f"\nProcessed: {ntsb_evidence['metadata']['event_id']}")
    print(f"  Hypothesis: {hypothesis}  ({hypothesis_reason})")
    print(f"  Belief: {final_belief}  (base: {base_confidence}, cross-source agreement: {agrees})")
    print(f"  Saved to: {output_path}")

print(f"\nDone. {len(ntsb_files)} fusion evidence object(s) generated in '{OUTPUT_DIR}/'.")