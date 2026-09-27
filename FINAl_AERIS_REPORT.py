# ============================================================
# AERIS Project — Phase 1 Investigation Report (HTML Presentation Layer)
# Reads the existing JSON output and renders a focused, presentable report.
# ============================================================

import json
import os

INPUT_PATH = "Reports/Phase1_Explainable_Investigation_Output.json"
OUTPUT_PATH = "Reports/Phase1_Investigation_Report.html"

with open(INPUT_PATH, "r") as f:
    report = json.load(f)

risk_level = report["aircraft_risk_level"]
confidence = report["overall_confidence_belief"]
subsystem = report["most_probable_fault_subsystem"]
cmapss = report["supporting_sensor_evidence_cmapss"]
ntsb = report["supporting_historical_investigation_ntsb"]
ngafid = report["supporting_flight_evidence_ngafid"]
fusion = report["fusion_transparency"]

RISK_COLORS = {
    "Low Risk": ("#1e7e34", "#d4edda"),
    "Medium Risk": ("#b8860b", "#fff3cd"),
    "High Risk": ("#a71d2a", "#f8d7da"),
}
risk_dark, risk_light = RISK_COLORS.get(risk_level, ("#333", "#eee"))

max_conflict = max(fusion["inter_source_conflict"].values())
conflict_note_html = ""
if max_conflict > 0.3:
    conflict_note_html = f"""
    <div class="note-box">
      <strong>Note on evidence sources:</strong> An inter-source conflict of
      {max_conflict:.2f} was detected during fusion, indicating the underlying
      evidence sources describe different aircraft cases in this Phase 1
      prototype. See the technical appendix for details.
    </div>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AERIS — Phase 1 Investigation Report</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f4f5f7;
    margin: 0;
    padding: 40px 20px;
    color: #222;
  }}
  .container {{ max-width: 880px; margin: 0 auto; }}
  .header {{ text-align: center; margin-bottom: 32px; }}
  .header h1 {{ font-size: 28px; margin-bottom: 4px; }}
  .header p {{ color: #666; font-size: 14px; }}

  .risk-banner {{
    background: {risk_light};
    border-left: 6px solid {risk_dark};
    border-radius: 8px;
    padding: 24px 28px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
  }}
  .risk-banner .risk-label {{ font-size: 14px; color: #555; text-transform: uppercase; letter-spacing: 1px; }}
  .risk-banner .risk-value {{ font-size: 32px; font-weight: 700; color: {risk_dark}; }}
  .risk-banner .confidence {{ text-align: right; }}
  .risk-banner .confidence .num {{ font-size: 26px; font-weight: 700; color: {risk_dark}; }}
  .risk-banner .confidence .lbl {{ font-size: 12px; color: #666; }}

  .subsystem-card {{
    background: white;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .subsystem-card .title {{ font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
  .subsystem-card .value {{ font-size: 22px; font-weight: 600; margin: 4px 0; }}
  .subsystem-card .source {{ font-size: 13px; color: #666; }}

  .evidence-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}
  .evidence-card {{
    background: white;
    border-radius: 8px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-top: 4px solid #444;
  }}
  .evidence-card.cmapss {{ border-top-color: #2e7d32; }}
  .evidence-card.ntsb {{ border-top-color: #8b1e3f; }}
  .evidence-card.ngafid {{ border-top-color: #1f6f8b; }}
  .evidence-card h3 {{ font-size: 15px; margin: 0 0 10px 0; }}
  .evidence-card .row {{ font-size: 13px; margin-bottom: 6px; display: flex; justify-content: space-between; gap: 8px; }}
  .evidence-card .row .k {{ color: #777; }}
  .evidence-card .row .v {{ font-weight: 600; text-align: right; }}
  .evidence-card .desc {{ font-size: 12.5px; color: #555; margin-top: 10px; line-height: 1.5; border-top: 1px solid #eee; padding-top: 10px; }}

  .note-box {{
    background: #fff8e1;
    border: 1px solid #f0d98c;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 13px;
    color: #6b5500;
    margin-bottom: 20px;
  }}

  .footer {{ text-align: center; font-size: 12px; color: #999; margin-top: 30px; }}
  .footer a {{ color: #888; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>AERIS — Phase 1 Investigation Report</h1>
    <p>Explainable Reasoning and Investigation System for Aerospace Maintenance</p>
    <p>Generated: {report['report_generated']}</p>
  </div>

  <div class="risk-banner">
    <div>
      <div class="risk-label">Aircraft Risk Level</div>
      <div class="risk-value">{risk_level}</div>
    </div>
    <div class="confidence">
      <div class="num">{confidence['pignistic_probability']*100:.1f}%</div>
      <div class="lbl">Overall Confidence (Belief: {confidence['belief_lower_bound']*100:.0f}%–{confidence['plausibility_upper_bound']*100:.0f}%)</div>
    </div>
  </div>

  <div class="subsystem-card">
    <div class="title">Most Probable Fault Subsystem</div>
    <div class="value">{subsystem['subsystem']}</div>
    <div class="source">Primary evidence source: {subsystem['primary_evidence_source']}</div>
  </div>

  {conflict_note_html}

  <div class="evidence-grid">

    <div class="evidence-card cmapss">
      <h3>🔧 CMAPSS — Sensor Evidence</h3>
      <div class="row"><span class="k">Predicted RUL</span><span class="v">{cmapss['predicted_rul_cycles']} cycles</span></div>
      <div class="row"><span class="k">Confidence</span><span class="v">{cmapss['prediction_confidence']*100:.1f}%</span></div>
      <div class="row"><span class="k">Inferred Subsystem</span><span class="v">{cmapss['inferred_subsystem']}</span></div>
      <div class="desc">Top contributing sensors (SHAP): {', '.join(cmapss['top_contributing_sensors_shap'])}</div>
    </div>

    <div class="evidence-card ntsb">
      <h3>📄 NTSB — Historical Investigation</h3>
      <div class="row"><span class="k">Matched Report</span><span class="v">{ntsb['matched_report_id']}</span></div>
      <div class="row"><span class="k">Aircraft</span><span class="v">{ntsb['aircraft_involved']}</span></div>
      <div class="row"><span class="k">Similarity</span><span class="v">{ntsb['semantic_similarity']*100:.1f}%</span></div>
      <div class="desc">{ntsb['probable_cause'][:180]}...</div>
    </div>

    <div class="evidence-card ngafid">
      <h3>✈️ NGAFID — Flight Evidence</h3>
      <div class="row"><span class="k">Failure Risk</span><span class="v">{ngafid['failure_risk_probability']*100:.1f}%</span></div>
      <div class="row"><span class="k">Diagnosed Subsystem</span><span class="v">{ngafid['diagnosed_subsystem']}</span></div>
      <div class="row"><span class="k">Diagnosis Confidence</span><span class="v">{ngafid['diagnosis_confidence']*100:.1f}%</span></div>
      <div class="desc">Top anomaly sensors: {', '.join(ngafid['top_anomaly_sensors'])}</div>
    </div>

  </div>

  <div class="footer">
    Full technical evidence objects and fusion mass distributions are available in the accompanying JSON appendix.<br>
    AERIS Phase 1 Prototype — Sources combined: {', '.join(fusion['sources_combined'])}
  </div>

</div>
</body>
</html>
"""

os.makedirs("Reports", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Saved presentable HTML report to: {OUTPUT_PATH}")
print("Open it in any browser to view.")