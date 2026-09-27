# AERIS — Phase 1 Streamlit Demo

This app has two tabs:

1. **Full AERIS Pipeline Demo** — loads your precomputed
   `Phase1_Explainable_Investigation_Output.json` (the real, end-to-end
   CMAPSS + NTSB + NGAFID + Dempster-Shafer result) and displays it as a
   clean dashboard.

2. **NTSB Semantic Retrieval Explorer** — a live sandbox where you can type
   *any* failure description and search your real NTSB FAISS index. This
   demonstrates that retrieval generalizes beyond the single
   High-Pressure-Compressor scenario that CMAPSS FD001 always produces.

## Setup

1. Copy `app.py` and `requirements.txt` into your **AERIS project root**
   (the same folder that contains `Reports/`, `NTSB_metadata.csv`,
   `NTSB_FAISS.index`, etc.) — the app expects those relative paths.

   If your files live elsewhere, edit the path constants near the top of
   `app.py`:
   ```python
   REPORT_JSON_PATH = "Reports/Phase1_Explainable_Investigation_Output.json"
   NTSB_METADATA_PATH = "NTSB_metadata.csv"
   NTSB_FAISS_INDEX_PATH = "NTSB_FAISS.index"
   ```

2. Install dependencies (ideally inside your existing `aeris_env` venv):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Your browser will open automatically at `http://localhost:8501`.

## Notes

- Tab 1 requires that you've already run your Phase 1 report-generation
  script (`19_Generate_...py` or equivalent) so that
  `Reports/Phase1_Explainable_Investigation_Output.json` exists.
- Tab 2 loads the Sentence Transformer model (`all-MiniLM-L6-v2`) and your
  FAISS index once, then caches them (`@st.cache_resource`) so repeated
  queries are fast after the first load.
- If you see a "Missing required file(s)" error, double-check you're
  running `streamlit run app.py` from the correct working directory, or
  update the path constants as described above.
