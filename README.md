# End-to-End Master Analytics Dashboard

Two ways to run the same analysis:
- **`app.py`** — an interactive Streamlit dashboard (8 tabs)
- **`Master_Analytics_Notebook.ipynb`** — the same analyses as a Google Colab notebook

## Streamlit dashboard tabs
1. **Descriptive & Diagnostic** — summary stats, missing values, cross-tabulation,
   correlation heatmap, distribution explorer, and a group-comparison/bias check
   (boxplots + group means, with an automatic flag if a group deviates >25% from
   the overall average).
2. **Anomaly Detection** — Isolation Forest with an adjustable contamination rate,
   scatter plot and donut chart of flagged outliers.
3. **Clustering & RFM** — RFM column mapping + 1–5 quintile scoring, K-Means with
   elbow method and silhouette scores (auto-suggests the best k), a cluster
   profile table (mean of each variable per cluster, to interpret what each
   segment represents), an interactive 3D cluster view, and a 2×2 grid comparing
   all four linkage methods (single, complete, average, Ward) side by side.
4. **Feature Engineering** — missing-value strategy (drop/mean/median), label
   encoding for categorical columns, IQR outlier capping, and a scaling preview.
   Click "Save as working dataset" and every other tab gets a checkbox to use
   this cleaned version instead of the raw upload.
5. **Classification** — KNN, Decision Tree, Random Forest, Gradient Boosting.
   Shows **train and test accuracy** (plus an "Overfit Gap" column), precision,
   recall, F1, ROC curves, and a selectable confusion matrix. Automatically
   warns if every model scores ~100% test accuracy (a sign of target leakage).
6. **Regression** — Linear, Ridge, Lasso with train/test R² and RMSE.
7. **Association Rule Mining** — Apriori + association rules (support,
   confidence, lift) over item/flag columns you select, with a lift bar chart.
8. **Prescriptive Insights** — auto-generated recommendations pulling together
   whatever you ran in tabs 1–7.

## Run the Streamlit app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the printed local URL (usually http://localhost:8501) and upload your
`.xlsx` or `.csv` from the sidebar.

## Deploy on Streamlit Community Cloud (share.streamlit.io)
1. Create a **public GitHub repository**.
2. Upload `app.py` and `requirements.txt` to the **root** of that repo.
3. Go to https://share.streamlit.io → **New app**.
4. Pick your repo/branch, set **Main file path** to `app.py`, click **Deploy**.
5. Once live, upload your dataset from the sidebar (it's uploaded at runtime,
   never committed to GitHub).

## Run the Colab notebook
1. Upload `Master_Analytics_Notebook.ipynb` to Google Drive, or go to
   https://colab.research.google.com → File → Upload notebook.
2. Runtime → Run all.
3. The second code cell opens a file-picker — upload your `.xlsx`/`.csv` there.
4. Each section (Anomaly Detection, Clustering, Classification, Regression,
   Association Rules) has a few `ALL_CAPS` variables near the top of its cell
   (e.g. `CLUSTER_VARS`, `TARGET_COL`, `BASKET_COLS`) — edit those to point at
   your actual column names, then re-run that cell.

## Notes on column requirements
- **Classification** needs a column with exactly 2 unique values to use as the target.
- **Association Rule Mining** needs 2+ item columns that are binary/Yes-No flags
  (e.g. `Gym_Bag`, `Towel`, `Pre_Workout`, `Disposable_Socks`).
- **RFM** fields are optional column mappings — pick whichever columns in your
  file represent Recency / Frequency / Monetary, or skip any you don't have.

## Common reasons a Streamlit Cloud deploy fails
- **`app.py` not found** — make sure it's at the repo root (or set the correct
  "Main file path" if it's in a subfolder).
- **Missing package errors** — every import in `app.py` has a matching line in
  `requirements.txt` already, so this shouldn't happen unless you add new imports.
- **A tab shows a warning instead of a chart** — it's telling you which column
  type it's missing (e.g. "need a binary target column"); pick the right
  columns in that tab's dropdowns/multiselects and it will render.
- **Classification crashes with a `ValueError` about precision/labels** — this
  was a real bug in an earlier version (a non-0/1 binary target crashed
  scikit-learn's precision/recall). It's fixed in this version: the target is
  always normalized to clean 0/1 codes before training, and the app shows you
  the mapping it used.
