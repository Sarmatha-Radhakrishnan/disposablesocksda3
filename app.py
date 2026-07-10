"""
End-to-End Master Analytics Dashboard (v2)
--------------------------------------------
Tabs:
  1. Descriptive & Diagnostic  — summary stats, cross-tabulation, group-bias exploration
  2. Anomaly Detection         — Isolation Forest
  3. Clustering & RFM          — K-Means (elbow/silhouette), 4-linkage dendrograms, 3D view, RFM
  4. Feature Engineering       — imputation, encoding, outlier capping (feeds later tabs)
  5. Classification            — KNN, Decision Tree, Random Forest, Gradient Boosting
  6. Regression                 — Linear, Ridge, Lasso
  7. Association Rule Mining    — Apriori / bundling
  8. Prescriptive Insights      — auto-summarized recommendations

Deploy on Streamlit Community Cloud: push this folder to a GitHub repo
containing app.py + requirements.txt, then point share.streamlit.io at it.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Master Analytics Dashboard", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 End-to-End Master Analytics Dashboard")
st.caption(
    "Descriptive & Diagnostic Analysis · Anomaly Detection · Clustering & RFM · "
    "Feature Engineering · Classification · Regression · Association Rule Mining · Prescriptive Insights"
)

# ----------------------------------------------------------------------------
# Sidebar — data input
# ----------------------------------------------------------------------------
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (.xlsx or .csv)", type=["xlsx", "xls", "csv"])

if "df" not in st.session_state:
    st.session_state.df = None
if "df_engineered" not in st.session_state:
    st.session_state.df_engineered = None
if "results" not in st.session_state:
    st.session_state.results = {}

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.session_state.df_engineered = None  # reset downstream state on new upload
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

df = st.session_state.df

if df is None:
    st.info(
        "👈 Upload an Excel (.xlsx) or CSV file from the sidebar to get started. "
        "All 8 tabs below will activate once data is loaded."
    )
    st.stop()

st.sidebar.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]
all_cols = df.columns.tolist()


def get_working_df(key_prefix, label="Use the engineered dataset from the Feature Engineering tab"):
    """Lets each analytical tab optionally use the cleaned/encoded dataframe."""
    if st.session_state.df_engineered is not None:
        use_eng = st.checkbox(label, value=True, key=f"{key_prefix}_use_eng")
        if use_eng:
            return st.session_state.df_engineered
    return df


tabs = st.tabs(
    [
        "📊 1. Desc & Diag",
        "🚨 2. Anomalies",
        "🧩 3. Clustering & RFM",
        "⚙️ 4. Feature Engineering",
        "🔮 5. Classification",
        "📈 6. Regression",
        "🔗 7. Association Rule Mining",
        "💡 8. Prescriptive",
    ]
)

# ============================================================================
# TAB 1 — DESCRIPTIVE & DIAGNOSTIC
# ============================================================================
with tabs[0]:
    st.header("Descriptive & Diagnostic Analysis")

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing.rename("Missing Count"), use_container_width=True)

    st.markdown("---")
    st.subheader("Cross-Tabulation")
    st.write("Compare two variables side by side. Numeric columns are auto-binned into quartiles.")
    c1, c2 = st.columns(2)
    ct_var1 = c1.selectbox("Row variable", all_cols, key="ct1")
    ct_var2 = c2.selectbox("Column variable", all_cols, index=min(1, len(all_cols) - 1), key="ct2")

    def to_categorical(series, n_bins=4):
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > n_bins:
            return pd.qcut(series, q=n_bins, duplicates="drop").astype(str)
        return series.astype(str)

    if ct_var1 and ct_var2:
        row = to_categorical(df[ct_var1])
        col = to_categorical(df[ct_var2])
        crosstab = pd.crosstab(row, col)
        st.dataframe(crosstab, use_container_width=True)
        fig_ct = px.imshow(crosstab, text_auto=True, color_continuous_scale="Teal",
                            title=f"Cross-tab: {ct_var1} × {ct_var2}")
        st.plotly_chart(fig_ct, use_container_width=True)

    st.markdown("---")
    if len(numeric_cols) >= 2:
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Teal",
                              title="Correlation Matrix (numeric fields)")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Distribution Explorer")
        dist_col = st.selectbox("Choose a numeric column", numeric_cols, key="dist_col")
        fig = px.histogram(df, x=dist_col, nbins=30, marginal="box", title=f"Distribution of {dist_col}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Diagnostic: Group Comparison & Bias Check")
    st.write(
        "Pick a numeric outcome and a grouping variable to check whether behavior is "
        "uneven across groups (e.g. does spend differ meaningfully by age band?)."
    )
    if numeric_cols and all_cols:
        c1, c2 = st.columns(2)
        diag_metric = c1.selectbox("Numeric outcome", numeric_cols, key="diag_metric")
        diag_group_raw = c2.selectbox("Group by", [c for c in all_cols if c != diag_metric], key="diag_group")
        group_series = to_categorical(df[diag_group_raw], n_bins=4)
        diag_df = pd.DataFrame({diag_metric: df[diag_metric], "Group": group_series}).dropna()

        group_stats = diag_df.groupby("Group")[diag_metric].agg(["count", "mean", "median", "std"]).reset_index()
        st.dataframe(group_stats, use_container_width=True)

        overall_mean = diag_df[diag_metric].mean()
        max_dev = (group_stats["mean"] - overall_mean).abs().max()
        rel_dev = max_dev / overall_mean if overall_mean else 0
        if rel_dev > 0.25:
            st.warning(
                f"⚠️ At least one group's average {diag_metric} deviates more than 25% "
                f"from the overall mean ({overall_mean:.2f}) — worth investigating whether "
                f"this reflects a real segment difference or a sampling/data-quality issue."
            )
        fig_box = px.box(diag_df, x="Group", y=diag_metric, color="Group",
                          title=f"{diag_metric} by {diag_group_raw}")
        st.plotly_chart(fig_box, use_container_width=True)

    st.session_state.results["n_rows"] = df.shape[0]

# ============================================================================
# TAB 2 — ANOMALY DETECTION
# ============================================================================
with tabs[1]:
    st.header("Anomaly Detection (Rare & Risky Events)")
    st.write(
        "Isolation Forest isolates points that are far from the bulk of the data — "
        "useful for flagging rare, risky, or high-value outlier behavior before it's "
        "averaged away by the rest of the analysis."
    )
    work_df = get_working_df("anomaly")
    work_numeric = work_df.select_dtypes(include=np.number).columns.tolist()

    if len(work_numeric) < 2:
        st.warning("Need at least 2 numeric columns to run anomaly detection.")
    else:
        anomaly_features = st.multiselect(
            "Select features for anomaly detection", work_numeric,
            default=work_numeric[: min(4, len(work_numeric))],
        )
        contamination = st.slider("Select Anomaly Contamination Rate (%)", 1, 25, 5) / 100.0

        if len(anomaly_features) >= 2:
            X = work_df[anomaly_features].dropna()
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(X)
            df_anom = work_df.loc[X.index].copy()
            df_anom["Anomaly"] = np.where(preds == -1, "Anomaly", "Normal")

            n_anom = int((preds == -1).sum())
            st.success(
                f"Detected {n_anom} anomalies out of {len(X)} rows "
                f"({n_anom / len(X):.1%}) at a {contamination:.0%} contamination rate."
            )

            c1, c2 = st.columns(2)
            xcol = c1.selectbox("X axis", anomaly_features, index=0, key="anom_x")
            ycol = c2.selectbox("Y axis", anomaly_features, index=min(1, len(anomaly_features) - 1), key="anom_y")
            fig = px.scatter(df_anom, x=xcol, y=ycol, color="Anomaly",
                              color_discrete_map={"Anomaly": "#B85042", "Normal": "#028090"},
                              title="Anomaly Detection Scatter")
            st.plotly_chart(fig, use_container_width=True)

            fig_donut = go.Figure(data=[go.Pie(
                labels=["Anomalies", "Typical"], values=[n_anom, len(X) - n_anom], hole=0.55,
                marker_colors=["#B85042", "#028090"],
            )])
            fig_donut.update_layout(title="Anomaly Rate")
            st.plotly_chart(fig_donut, use_container_width=True)

            st.session_state.results["n_anomalies"] = n_anom
            st.session_state.results["anomaly_rate"] = n_anom / len(X)
            st.session_state.results["anomaly_sample"] = len(X)
        else:
            st.info("Select at least 2 features above.")

# ============================================================================
# TAB 3 — CLUSTERING & RFM
# ============================================================================
with tabs[2]:
    st.header("Clustering & RFM Analysis")
    work_df = get_working_df("cluster")
    work_numeric = work_df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("1. RFM Synthesis")
    st.write("Map columns from your data to Recency / Frequency / Monetary, or skip if not applicable.")
    c1, c2, c3 = st.columns(3)
    rec_col = c1.selectbox("Recency column", ["(none)"] + all_cols, key="rfm_r")
    freq_col = c2.selectbox("Frequency column", ["(none)"] + all_cols, key="rfm_f")
    mon_col = c3.selectbox("Monetary column", ["(none)"] + all_cols, key="rfm_m")
    rfm_cols = [c for c in [rec_col, freq_col, mon_col] if c != "(none)"]

    if rfm_cols:
        rfm_view = df[rfm_cols].copy()
        rename_map = {}
        if rec_col != "(none)":
            rename_map[rec_col] = "Recency"
        if freq_col != "(none)":
            rename_map[freq_col] = "Frequency"
        if mon_col != "(none)":
            rename_map[mon_col] = "Monetary"
        rfm_view = rfm_view.rename(columns=rename_map)

        # simple 1-5 RFM scoring: lower recency = better (5), higher freq/monetary = better (5)
        rfm_scored = rfm_view.copy()
        if "Recency" in rfm_scored.columns:
            rfm_scored["R_Score"] = pd.qcut(rfm_scored["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
        if "Frequency" in rfm_scored.columns:
            rfm_scored["F_Score"] = pd.qcut(rfm_scored["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        if "Monetary" in rfm_scored.columns:
            rfm_scored["M_Score"] = pd.qcut(rfm_scored["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        score_cols = [c for c in ["R_Score", "F_Score", "M_Score"] if c in rfm_scored.columns]
        if score_cols:
            rfm_scored["RFM_Total"] = rfm_scored[score_cols].sum(axis=1)
        st.dataframe(rfm_scored.head(15), use_container_width=True)
        st.caption("R/F/M scores are 1 (worst quintile) to 5 (best quintile); RFM_Total sums whichever scores are available.")
    else:
        st.info("Pick at least one RFM-style column above to see the scored table.")

    st.markdown("---")
    st.subheader("2. Variable Selection & K-Means Clustering")
    cluster_vars = st.multiselect(
        "Select 3-5 variables for clustering", work_numeric,
        default=work_numeric[: min(4, len(work_numeric))],
    )

    if len(cluster_vars) >= 2:
        X = work_df[cluster_vars].dropna()
        X_scaled = StandardScaler().fit_transform(X)

        inertias, sils, valid_ks = [], [], []
        max_k = min(7, len(X) - 1)
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X_scaled, labels))
            valid_ks.append(k)

        best_k_by_sil = valid_ks[int(np.argmax(sils))]

        c1, c2 = st.columns(2)
        with c1:
            fig_elbow = px.line(x=valid_ks, y=inertias, markers=True,
                                 labels={"x": "k", "y": "Inertia"}, title="Elbow Method")
            st.plotly_chart(fig_elbow, use_container_width=True)
        with c2:
            fig_sil = px.line(x=valid_ks, y=sils, markers=True,
                               labels={"x": "k", "y": "Silhouette Score"}, title="Silhouette Scores")
            st.plotly_chart(fig_sil, use_container_width=True)

        st.info(f"💡 Highest silhouette score is at **k = {best_k_by_sil}** ({max(sils):.3f}) — "
                f"a good starting point for the 'optimal' cluster count, though business "
                f"interpretability can justify choosing a nearby k instead.")

        k_opt = st.slider("Select Number of Clusters to Use", min(valid_ks), max(valid_ks), value=best_k_by_sil)
        km_final = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
        cluster_labels = km_final.fit_predict(X_scaled)
        df_clustered = work_df.loc[X.index].copy()
        df_clustered["Cluster"] = cluster_labels.astype(str)

        st.subheader("3. 3D Cluster Visualization")
        if len(cluster_vars) >= 3:
            v1, v2, v3 = st.columns(3)
            ax1 = v1.selectbox("X axis", cluster_vars, index=0, key="c3d_x")
            ax2 = v2.selectbox("Y axis", cluster_vars, index=1, key="c3d_y")
            ax3 = v3.selectbox("Z axis", cluster_vars, index=2, key="c3d_z")
            fig3d = px.scatter_3d(df_clustered, x=ax1, y=ax2, z=ax3, color="Cluster",
                                   opacity=0.75, title="3D Cluster View")
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            fig2d = px.scatter(df_clustered, x=cluster_vars[0], y=cluster_vars[1],
                                color="Cluster", title="2D Cluster View")
            st.plotly_chart(fig2d, use_container_width=True)

        st.subheader("Cluster Profiles — What Pattern Does Each Cluster Represent?")
        profile = df_clustered.groupby("Cluster")[cluster_vars].mean().round(2)
        profile["Size"] = df_clustered["Cluster"].value_counts()
        st.dataframe(profile, use_container_width=True)
        st.caption(
            "Compare each cluster's row to the others: the variable(s) where a cluster "
            "sits noticeably above or below its peers is what defines that segment's persona."
        )

        st.markdown("---")
        st.subheader("4. Hierarchical Clustering — Linkage Comparison")
        st.write(
            "Different linkage rules define cluster distance differently, which changes "
            "how the dendrogram groups points: **Single** (nearest point), **Complete** "
            "(farthest point), **Average** (mean distance), **Ward** (minimizes within-cluster variance)."
        )
        import matplotlib.pyplot as plt

        sample_for_dendro = X_scaled[: min(150, len(X_scaled))]
        linkage_methods = ["single", "complete", "average", "ward"]
        fig_dendro, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, method in zip(axes.ravel(), linkage_methods):
            Z = linkage(sample_for_dendro, method=method)
            dendrogram(Z, ax=ax, color_threshold=0.7 * max(Z[:, 2]), no_labels=True)
            ax.set_title(f"{method.capitalize()} Linkage")
            ax.set_ylabel("Distance")
        plt.tight_layout()
        st.pyplot(fig_dendro)

        st.caption(
            "Ward linkage is the most common default for numeric business data (it tends to "
            "produce compact, evenly sized clusters, mirroring K-Means). Single linkage is "
            "the most sensitive to outliers/chaining and is shown here mainly for contrast."
        )

        st.session_state.results["k_opt"] = k_opt
        st.session_state.results["cluster_vars"] = cluster_vars
        st.session_state.results["cluster_sizes"] = df_clustered["Cluster"].value_counts().to_dict()
        st.session_state.results["best_k_by_sil"] = best_k_by_sil
    else:
        st.info("Select at least 2 numeric variables to cluster.")

# ============================================================================
# TAB 4 — FEATURE ENGINEERING
# ============================================================================
with tabs[3]:
    st.header("Feature Engineering")
    st.write(
        "Configure preprocessing here, then check **'Use the engineered dataset'** in any "
        "other tab to apply it there. If you skip this tab, every other tab just uses your "
        "raw uploaded data (with its own internal scaling where needed)."
    )

    eng_df = df.copy()

    st.subheader("1. Missing Value Handling")
    na_strategy = st.radio(
        "Strategy for numeric missing values",
        ["Leave as-is", "Drop rows with any missing values", "Impute with mean", "Impute with median"],
        horizontal=True,
    )
    num_cols_eng = eng_df.select_dtypes(include=np.number).columns.tolist()
    if na_strategy == "Drop rows with any missing values":
        eng_df = eng_df.dropna()
    elif na_strategy == "Impute with mean":
        eng_df[num_cols_eng] = eng_df[num_cols_eng].fillna(eng_df[num_cols_eng].mean())
    elif na_strategy == "Impute with median":
        eng_df[num_cols_eng] = eng_df[num_cols_eng].fillna(eng_df[num_cols_eng].median())

    st.subheader("2. Categorical Encoding")
    cat_cols_eng = eng_df.select_dtypes(exclude=np.number).columns.tolist()
    encode_cols = st.multiselect("Label-encode these columns", cat_cols_eng, default=cat_cols_eng)
    for c in encode_cols:
        eng_df[c] = LabelEncoder().fit_transform(eng_df[c].astype(str))

    st.subheader("3. Outlier Capping (IQR method)")
    cap_outliers = st.checkbox("Cap numeric outliers at 1.5×IQR (winsorize)", value=False)
    if cap_outliers:
        for c in num_cols_eng:
            q1, q3 = eng_df[c].quantile(0.25), eng_df[c].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            eng_df[c] = eng_df[c].clip(lower, upper)

    st.subheader("4. Scaling Preview (for reference — modeling tabs scale internally)")
    scale_method = st.selectbox("Preview scaling method", ["None", "StandardScaler (z-score)", "MinMaxScaler (0-1)"])
    preview_df = eng_df.copy()
    if scale_method != "None":
        num_cols_now = preview_df.select_dtypes(include=np.number).columns.tolist()
        scaler = StandardScaler() if "Standard" in scale_method else MinMaxScaler()
        preview_df[num_cols_now] = scaler.fit_transform(preview_df[num_cols_now])

    st.subheader("Engineered Data Preview")
    st.dataframe(preview_df.head(15), use_container_width=True)

    if st.button("💾 Save this as the working dataset for other tabs"):
        st.session_state.df_engineered = eng_df
        st.success(
            f"Saved. Engineered dataset has {eng_df.shape[0]} rows × {eng_df.shape[1]} columns. "
            f"Go to any other tab and check 'Use the engineered dataset' to apply it."
        )

# ============================================================================
# TAB 5 — CLASSIFICATION
# ============================================================================
with tabs[4]:
    st.header("Predictive Classification Models")
    work_df = get_working_df("clf")
    work_numeric = work_df.select_dtypes(include=np.number).columns.tolist()

    binary_candidates = [c for c in work_df.columns if work_df[c].nunique() == 2]
    if not binary_candidates:
        st.warning("No binary (2-value) column found for a classification target. "
                   "Add one to your dataset (e.g. Purchased Y/N) to use this tab.")
    else:
        target_col = st.selectbox("Select target (binary) column", binary_candidates)
        feature_options = [c for c in work_numeric if c != target_col]
        feature_cols = st.multiselect(
            "Select feature columns", feature_options,
            default=feature_options[: min(5, len(feature_options))],
        )

        if len(feature_cols) >= 1:
            data = work_df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y_raw = data[target_col]

            uniques = sorted(y_raw.unique().tolist(), key=lambda v: str(v))
            if len(uniques) != 2:
                st.error(f"'{target_col}' has {len(uniques)} unique value(s) after removing "
                         f"missing rows — a classification target needs exactly 2.")
                st.stop()
            label_map = {uniques[0]: 0, uniques[1]: 1}
            y = y_raw.map(label_map)
            st.caption(f"Target encoding used: {uniques[0]} → 0, {uniques[1]} → 1")

            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100.0
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() > 1 else None
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            models = {
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }

            rows, roc_data, cms = [], {}, {}
            for name, model in models.items():
                model.fit(X_train_s, y_train)
                train_pred = model.predict(X_train_s)
                pred = model.predict(X_test_s)
                proba = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else pred
                rows.append({
                    "Model": name,
                    "Train Accuracy": accuracy_score(y_train, train_pred),
                    "Test Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, zero_division=0),
                    "Recall": recall_score(y_test, pred, zero_division=0),
                    "F1-Score": f1_score(y_test, pred, zero_division=0),
                })
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_data[name] = (fpr, tpr, auc(fpr, tpr))
                cms[name] = confusion_matrix(y_test, pred)

            results_df = pd.DataFrame(rows)
            results_df["Overfit Gap"] = (results_df["Train Accuracy"] - results_df["Test Accuracy"]).round(3)
            st.dataframe(
                results_df.style.background_gradient(cmap="YlGn", subset=["Test Accuracy", "F1-Score"]),
                use_container_width=True,
            )
            st.caption("Overfit Gap = Train Accuracy − Test Accuracy. A large positive gap means the "
                       "model memorized training data rather than learning a generalizable pattern.")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("ROC Curves")
                fig_roc = go.Figure()
                for name, (fpr, tpr, auc_val) in roc_data.items():
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc_val:.2f})"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Chance"))
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)
            with c2:
                st.subheader("Confusion Matrix")
                cm_model = st.selectbox("Select model for Confusion Matrix", list(models.keys()))
                cm = cms[cm_model]
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                                    labels=dict(x="Predicted", y="Actual"), title=f"{cm_model} Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)

            if results_df["Test Accuracy"].min() >= 0.999:
                st.warning(
                    "⚠️ All models are scoring at or near 100% test accuracy. This is a strong "
                    "signal of possible **target leakage** (a feature that encodes the target) "
                    "or a trivially separable target — verify your feature set before trusting "
                    "this result for a business decision."
                )

            st.session_state.results["clf_table"] = results_df
            st.session_state.results["clf_perfect_flag"] = bool(results_df["Test Accuracy"].min() >= 0.999)
        else:
            st.info("Select at least 1 feature column.")

# ============================================================================
# TAB 6 — REGRESSION
# ============================================================================
with tabs[5]:
    st.header("Predictive Regression")
    work_df = get_working_df("reg")
    work_numeric = work_df.select_dtypes(include=np.number).columns.tolist()

    reg_target = st.selectbox("Select target (continuous) column", work_numeric, key="reg_target")
    reg_feature_options = [c for c in work_numeric if c != reg_target]
    reg_features = st.multiselect(
        "Select feature columns", reg_feature_options,
        default=reg_feature_options[: min(5, len(reg_feature_options))], key="reg_features",
    )

    if len(reg_features) >= 1:
        data = work_df[reg_features + [reg_target]].dropna()
        X = data[reg_features]
        y = data[reg_target]

        test_size = st.slider("Test set size (%)", 10, 40, 20, key="reg_test_size") / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg_models = {"Linear": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso()}
        rows = []
        for name, model in reg_models.items():
            model.fit(X_train_s, y_train)
            train_pred = model.predict(X_train_s)
            pred = model.predict(X_test_s)
            rows.append({
                "Model": name,
                "Train R²": r2_score(y_train, train_pred),
                "Test R²": r2_score(y_test, pred),
                "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            })
        reg_results = pd.DataFrame(rows)
        best_idx = reg_results["Test R²"].idxmax()
        st.dataframe(reg_results.style.highlight_max(subset=["Test R²"], color="#FFF6C9"), use_container_width=True)

        fig_bar = px.bar(reg_results, x="Model", y="Test R²", color="Model", title="Model Fit (Test R²)", text_auto=".3f")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.session_state.results["reg_table"] = reg_results
        st.session_state.results["reg_best_model"] = reg_results.loc[best_idx, "Model"]
    else:
        st.info("Select at least 1 feature column.")

# ============================================================================
# TAB 7 — ASSOCIATION RULE MINING
# ============================================================================
with tabs[6]:
    st.header("Association Rule Mining (Bundling Analysis)")
    st.info("Select item / product columns that indicate purchase (0/1, or Yes/No, or a low-cardinality flag).")

    binary_like_cols = [c for c in all_cols if df[c].nunique() <= 3]
    basket_cols = st.multiselect("Select item columns for basket analysis", binary_like_cols)

    if len(basket_cols) >= 2:
        basket_df = df[basket_cols].copy()
        for c in basket_cols:
            if basket_df[c].dtype == object:
                basket_df[c] = basket_df[c].astype(str).str.strip().str.lower().isin(["1", "yes", "y", "true"])
            else:
                basket_df[c] = basket_df[c].astype(bool)

        min_support = st.slider("Minimum support", 0.01, 0.5, 0.05, step=0.01)
        frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            st.warning("No frequent itemsets found at this support threshold. Try lowering it.")
        else:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            rules = rules.sort_values("lift", ascending=False)
            if rules.empty:
                st.warning("No association rules found above lift = 1.0.")
            else:
                top_n = st.slider("Show top N rules (by lift)", 3, 20, 4)
                display_rules = rules.head(top_n).copy()
                display_rules["antecedents"] = display_rules["antecedents"].apply(lambda s: ", ".join(list(s)))
                display_rules["consequents"] = display_rules["consequents"].apply(lambda s: ", ".join(list(s)))
                st.dataframe(
                    display_rules[["antecedents", "consequents", "support", "confidence", "lift"]]
                    .style.format({"support": "{:.3f}", "confidence": "{:.3f}", "lift": "{:.4f}"}),
                    use_container_width=True,
                )
                fig_lift = px.bar(display_rules, x=display_rules.index.astype(str), y="lift",
                                   title="Association Rule Lift", labels={"x": "Rule"})
                st.plotly_chart(fig_lift, use_container_width=True)
                st.session_state.results["basket_rules"] = display_rules
    else:
        st.info("Select at least 2 item columns above to run association rule mining.")

# ============================================================================
# TAB 8 — PRESCRIPTIVE
# ============================================================================
with tabs[7]:
    st.header("Prescriptive Recommendations")
    st.subheader("Actionable Insights Auto-Summarized From the Tabs Above")

    r = st.session_state.results
    insights = []

    if "cluster_sizes" in r:
        insights.append(
            f"**Targeted Segmentation:** K-Means with k={r.get('k_opt')} on "
            f"{', '.join(r.get('cluster_vars', []))} produced clusters sized {r['cluster_sizes']} "
            f"(silhouette-optimal k was {r.get('best_k_by_sil')}). Use the cluster profile table "
            f"in Tab 3 to name each segment and target it separately."
        )
    if "n_anomalies" in r:
        insights.append(
            f"**Anomaly Constraints:** {r['n_anomalies']} of {r.get('anomaly_sample', '?')} rows "
            f"({r.get('anomaly_rate', 0):.1%}) were flagged as outliers. Consider a dedicated "
            f"premium/retention track for this cohort rather than filtering them out."
        )
    if "basket_rules" in r and not r["basket_rules"].empty:
        top_rule = r["basket_rules"].iloc[0]
        insights.append(
            f"**Bundling Strategy:** The strongest rule ({top_rule['antecedents']} → "
            f"{top_rule['consequents']}) has lift {top_rule['lift']:.2f} and confidence "
            f"{top_rule['confidence']:.0%}. Start with this bundle before expanding further."
        )
    if "clf_table" in r:
        best_clf = r["clf_table"].sort_values("F1-Score", ascending=False).iloc[0]
        note = " ⚠️ Validate for leakage/overfitting before deploying." if r.get("clf_perfect_flag") else ""
        insights.append(
            f"**Model Deployment:** {best_clf['Model']} gave the best F1-Score "
            f"({best_clf['F1-Score']:.3f}) with an overfit gap of {best_clf['Overfit Gap']:.3f}.{note}"
        )
    if "reg_table" in r:
        insights.append(
            f"**Spend Forecasting:** {r['reg_best_model']} was the strongest regressor on the "
            f"held-out test set — use it as the basis for spend/pricing projections."
        )

    if not insights:
        st.info("Run the analyses in the tabs above first — recommendations will populate here automatically.")
    else:
        for i, insight in enumerate(insights, start=1):
            st.markdown(f"{i}. {insight}")
