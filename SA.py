"""
╔══════════════════════════════════════════════════════════════════════╗
║   Mall Customer Segmentation Dashboard                               ║
║   PCA + Simulated Annealing + K-Means  |  Interactive Streamlit App  ║
╚══════════════════════════════════════════════════════════════════════╝
Jalankan dengan:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import math, random, time, itertools, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)

# ──────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Segmentation · SA+KMeans",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
#  GLOBAL THEME
# ──────────────────────────────────────────────────────────────────────────
PAL    = ["#5C6BC0","#26A69A","#EF5350","#FFA726","#66BB6A","#AB47BC","#EC407A"]
BG     = "#0F1117"
CARD   = "#1A1D2B"
GRID_C = "#2C2F3E"
FG     = "#E8EAF6"
MUTED  = "#7986CB"
ACCENT = "#FFA726"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color=FG, family="Sora, DM Sans, sans-serif"),
    xaxis=dict(gridcolor=GRID_C, zerolinecolor=GRID_C),
    yaxis=dict(gridcolor=GRID_C, zerolinecolor=GRID_C),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor=CARD, bordercolor=GRID_C, borderwidth=1),
)

# ──────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {{
  --bg: {BG}; --card: {CARD}; --accent: {ACCENT};
  --fg: {FG}; --muted: {MUTED}; --grid: {GRID_C};
}}

/* Root background */
.stApp {{ background: var(--bg) !important; }}
section[data-testid="stSidebar"] {{ background: #12151F !important; border-right: 1px solid var(--grid); }}
section[data-testid="stSidebar"] * {{ color: var(--fg) !important; }}

/* Typography */
body, p, div, span {{ font-family: 'Sora', sans-serif !important; color: var(--fg); }}
h1, h2, h3 {{ font-family: 'Sora', sans-serif !important; font-weight: 700; color: var(--fg); }}
code, pre {{ font-family: 'DM Mono', monospace !important; }}

/* Metric cards */
.metric-card {{
  background: var(--card);
  border: 1px solid var(--grid);
  border-radius: 12px;
  padding: 20px 24px;
  text-align: center;
  transition: transform .2s, border-color .2s;
}}
.metric-card:hover {{ transform: translateY(-3px); border-color: var(--accent); }}
.metric-label {{ font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }}
.metric-value {{ font-size: 30px; font-weight: 700; color: var(--fg); }}
.metric-delta {{ font-size: 12px; margin-top: 4px; }}

/* Section header */
.section-header {{
  display: flex; align-items: center; gap: 12px;
  padding: 14px 20px;
  background: linear-gradient(90deg, {CARD}, transparent);
  border-left: 3px solid {ACCENT};
  border-radius: 0 8px 8px 0;
  margin: 24px 0 16px;
}}
.section-header h3 {{ margin: 0; font-size: 16px; letter-spacing: .5px; }}

/* Cluster badge */
.cluster-badge {{
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--card); border: 1px solid var(--grid);
  border-radius: 20px; padding: 6px 16px;
  font-size: 13px; font-weight: 600;
}}

/* Tag */
.tag {{
  display: inline-block; background: rgba(255,167,38,.15);
  color: {ACCENT}; border: 1px solid rgba(255,167,38,.3);
  border-radius: 6px; padding: 2px 10px; font-size: 11px;
  font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
}}

/* Tab styling */
[data-testid="stTab"] > div {{ color: var(--muted); }}

/* Plotly chart containers */
[data-testid="stPlotlyChart"] {{ background: transparent !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }} 
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--grid); border-radius: 3px; }}

/* DataFrames */
[data-testid="stDataFrame"] {{ background: var(--card) !important; }}

/* Spinner */
.stSpinner > div {{ border-top-color: var(--accent) !important; }}

/* Slider */
.stSlider [data-testid="stThumbValue"] {{ color: var(--accent) !important; }}

/* Success/info boxes */
.stSuccess, .stInfo {{
  background: rgba(38,166,154,.1) !important;
  border-color: #26A69A !important;
}}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────
def section_header(icon, title):
    st.markdown(f"""
    <div class="section-header">
      <span style="font-size:20px">{icon}</span>
      <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value, delta, delta_color) in zip(cols, items):
        delta_html = (
            f'<div class="metric-delta" style="color:{delta_color}">{delta}</div>'
            if delta else ""
        )
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          {delta_html}
        </div>
        """, unsafe_allow_html=True)


def apply_layout(fig, title="", height=400):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=14, color=FG)), height=height)
    return fig


# ──────────────────────────────────────────────────────────────────────────
#  SA ENGINE
# ──────────────────────────────────────────────────────────────────────────
def run_sa(X, K, T0=100.0, alpha=0.95, T_min=1e-3,
           max_iter=600, sigma=0.5, seed=42):
    np.random.seed(seed); random.seed(seed)

    def eval_wcss(centers):
        km = KMeans(n_clusters=K, init=centers, n_init=1, max_iter=300, random_state=seed)
        km.fit(X)
        return km.inertia_, km.labels_, km.cluster_centers_

    idx0 = np.random.choice(len(X), K, replace=False)
    c_curr = X[idx0].copy()
    wcss_curr, lbl_curr, c_curr = eval_wcss(c_curr)
    c_best, wcss_best = c_curr.copy(), wcss_curr
    wcss_init = wcss_curr

    T = T0
    history, t_log, wcss_log, wcss_best_log = [], [T], [wcss_curr], [wcss_best]
    n_accept_better = n_accept_metro = n_reject = n_escape = 0
    prev_was_worse = False
    it = 0
    t0 = time.time()

    while T > T_min and it < max_iter:
        it += 1
        c_cand = c_curr + np.random.normal(0, sigma, c_curr.shape)
        wcss_cand, lbl_cand, c_cand = eval_wcss(c_cand)
        delta = wcss_cand - wcss_curr

        if delta < 0:
            prob = 1.0; accept = True; n_accept_better += 1
        else:
            prob = math.exp(-delta / T)
            accept = (random.random() < prob)
            if accept: n_accept_metro += 1
            else: n_reject += 1

        if accept:
            c_curr, wcss_curr = c_cand.copy(), wcss_cand
            if prev_was_worse and wcss_curr < wcss_best: n_escape += 1
            prev_was_worse = (delta >= 0)
        else:
            prev_was_worse = False

        if wcss_curr < wcss_best:
            wcss_best, c_best = wcss_curr, c_curr.copy()

        history.append(dict(iter=it, T=round(T,8), wcss_curr=round(wcss_curr,4),
                            wcss_best=round(wcss_best,4), delta=round(delta,4),
                            prob=round(prob,6),
                            status="Accept" if accept else "Reject"))
        t_log.append(T); wcss_log.append(wcss_curr); wcss_best_log.append(wcss_best)
        T *= alpha

    elapsed = time.time() - t0
    wcss_final, lbl_final, c_final = eval_wcss(c_best)
    total = n_accept_better + n_accept_metro + n_reject

    return dict(
        wcss_init=wcss_init, wcss_best_sa=wcss_best,
        wcss_final=wcss_final, labels=lbl_final, centers=c_final,
        hist=pd.DataFrame(history),
        t_log=t_log, wcss_log=wcss_log, wcss_best_log=wcss_best_log,
        iters=it, time=elapsed,
        accept_rate=(n_accept_better+n_accept_metro)/total*100,
        n_metro=n_accept_metro, n_better=n_accept_better,
        n_reject=n_reject, n_escape=n_escape,
        sil=silhouette_score(X, lbl_final),
        db=davies_bouldin_score(X, lbl_final),
        ch=calinski_harabasz_score(X, lbl_final),
        T0=T0, alpha=alpha, sigma=sigma, max_iter=max_iter,
    )


# ──────────────────────────────────────────────────────────────────────────
#  CACHED PIPELINE
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(uploaded_bytes, sep=";"):
    import io
    df = pd.read_csv(io.BytesIO(uploaded_bytes), sep=sep)
    le = LabelEncoder()
    df["Genre_enc"] = le.fit_transform(df["Genre"])
    FEATURES = ["Genre_enc","Age","Annual Income (k$)","Spending Score (1-100)"]
    X_raw = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return df, X_scaled, scaler, FEATURES


@st.cache_data
def run_pca(X_scaled, n_components=3):
    pca_full = PCA().fit(X_scaled)
    EV = pca_full.explained_variance_ratio_
    EV_CUM = np.cumsum(EV)
    pca3 = PCA(n_components=n_components).fit(X_scaled)
    pca2 = PCA(n_components=2).fit(X_scaled)
    X3 = pca3.transform(X_scaled)
    X2 = pca2.transform(X_scaled)
    return pca_full, pca3, pca2, X3, X2, EV, EV_CUM


@st.cache_data
def find_optimal_k(X3, k_max=10):
    rows = []
    for k in range(2, k_max+1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42).fit(X3)
        lbl = km.labels_
        rows.append(dict(K=k, WCSS=km.inertia_,
                         Silhouette=silhouette_score(X3, lbl),
                         Davies_Bouldin=davies_bouldin_score(X3, lbl),
                         Calinski_H=calinski_harabasz_score(X3, lbl)))
    df_k = pd.DataFrame(rows).set_index("K")
    K_opt = int(df_k["Silhouette"].idxmax())
    return df_k, K_opt


@st.cache_data
def get_baselines(X3, K):
    km_rand = KMeans(n_clusters=K, init="random",    n_init=30, random_state=42).fit(X3)
    km_kpp  = KMeans(n_clusters=K, init="k-means++", n_init=30, random_state=42).fit(X3)
    return km_rand, km_kpp


@st.cache_data
def run_sa_cached(X3_bytes, K, T0, alpha, T_min, max_iter, sigma, seed):
    X3 = np.frombuffer(X3_bytes).reshape(-1, K if K > 0 else 3)
    return run_sa(X3, K, T0=T0, alpha=alpha, T_min=T_min,
                  max_iter=max_iter, sigma=sigma, seed=seed)


@st.cache_data
def run_sensitivity(X3_bytes, K, ALPHAS, SIGMAS, T0, T_min, max_iter):
    X3 = np.frombuffer(X3_bytes).reshape(-1, 3)
    SENS = []
    for alpha, sigma in itertools.product(ALPHAS, SIGMAS):
        r = run_sa(X3, K, T0=T0, alpha=alpha, T_min=T_min,
                   max_iter=max_iter, sigma=sigma, seed=42)
        r.update({"alpha": alpha, "sigma": sigma})
        SENS.append(r)
    return SENS


# ──────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px; text-align:center'>
      <div style='font-size:32px'>🧠</div>
      <div style='font-size:16px;font-weight:700;letter-spacing:1px'>Mall Segmentation</div>
      <div style='font-size:11px;color:#7986CB;margin-top:4px'>PCA · SA · K-Means</div>
    </div>
    <hr style='border-color:#2C2F3E;margin:12px 0'>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📂 Upload Mall_Customers.csv", type=["csv"])
    csv_sep  = st.selectbox("Separator", [";", ",", "\\t"], index=0)

    st.markdown("---")
    st.markdown("**⚙️ PCA Settings**")
    n_pca = st.slider("PCA Components (Clustering)", 2, 4, 3)
    k_max = st.slider("Max K (Elbow Search)", 5, 15, 10)

    st.markdown("---")
    st.markdown("**🔥 Simulated Annealing**")
    T0    = st.slider("T₀  (Suhu Awal)",     10.0, 200.0, 100.0, 10.0)
    alpha = st.slider("α   (Cooling Rate)",   0.80, 0.99,  0.95,  0.01)
    sigma = st.slider("σ   (Perturbasi Std)", 0.1,  1.5,   0.5,   0.1)
    max_iter_sa = st.slider("Max Iterasi SA",  200, 1000, 600, 100)

    st.markdown("---")
    st.markdown("**📊 Sensitivity Grid**")
    alphas_str = st.text_input("α values (comma)", "0.85,0.92,0.97")
    sigmas_str = st.text_input("σ values (comma)", "0.3,0.5,0.8")

    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary")


# ──────────────────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='
  background:linear-gradient(135deg,#1A1D2B 0%,#0F1117 100%);
  border:1px solid #2C2F3E; border-radius:16px;
  padding:32px 36px 28px; margin-bottom:28px;
'>
  <div style='display:flex;align-items:center;gap:16px;margin-bottom:8px'>
    <span style='font-size:40px'>🏬</span>
    <div>
      <h1 style='margin:0;font-size:26px;letter-spacing:-.5px'>
        Mall Customer Segmentation Dashboard
      </h1>
      <p style='margin:4px 0 0;color:#7986CB;font-size:14px'>
        Principal Component Analysis  ·  Simulated Annealing  ·  K-Means Clustering
      </p>
    </div>
  </div>
  <div style='display:flex;gap:8px;margin-top:16px;flex-wrap:wrap'>
    <span class="tag">PCA</span>
    <span class="tag">K-Means</span>
    <span class="tag">Simulated Annealing</span>
    <span class="tag">Plotly Interactive</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
#  GUARD: Need file
# ──────────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("⬅️  Upload file **Mall_Customers.csv** di sidebar untuk memulai analisis.")
    st.markdown("""
    <div style='background:#1A1D2B;border:1px dashed #2C2F3E;border-radius:12px;padding:24px;margin-top:16px'>
      <h4 style='margin:0 0 12px'>📋 Format Data yang Diperlukan</h4>
      <p style='color:#7986CB'>Kolom wajib dalam file CSV:</p>
      <ul style='color:#7986CB'>
        <li><code>CustomerID</code></li>
        <li><code>Genre</code> (Male/Female)</li>
        <li><code>Age</code></li>
        <li><code>Annual Income (k$)</code></li>
        <li><code>Spending Score (1-100)</code></li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ──────────────────────────────────────────────────────────────────────────
raw_bytes = uploaded.read()
sep_map = {";": ";", ",": ",", "\\t": "\t"}
df, X_scaled, scaler, FEATURES = load_and_preprocess(raw_bytes, sep=sep_map[csv_sep])
NUM_COLS = ["Age","Annual Income (k$)","Spending Score (1-100)"]

# PCA
pca_full, pca3, pca2, X3, X2, EV, EV_CUM = run_pca(X_scaled, n_components=n_pca)

# Optimal K
df_k, K_opt = find_optimal_k(X3, k_max=k_max)

# Baselines
km_rand, km_kpp = get_baselines(X3, K_opt)
WCSS_BASE = km_kpp.inertia_
SIL_BASE  = silhouette_score(X3, km_kpp.labels_)
DB_BASE   = davies_bouldin_score(X3, km_kpp.labels_)
CH_BASE   = calinski_harabasz_score(X3, km_kpp.labels_)

# SA
with st.spinner("⚡ Menjalankan Simulated Annealing..."):
    X3_bytes = X3.tobytes()
    RES = run_sa_cached(X3_bytes, K_opt, T0, alpha, 1e-3, max_iter_sa, sigma, 42)

df["Cluster"] = RES["labels"]
cluster_ids   = sorted(df["Cluster"].unique())
profile = df.groupby("Cluster")[NUM_COLS].mean().round(2)
profile["N"] = df.groupby("Cluster").size()
profile["%"] = (profile["N"] / len(df) * 100).round(1)

# SA sensitivity
try:
    ALPHAS_S = [float(x.strip()) for x in alphas_str.split(",")]
    SIGMAS_S = [float(x.strip()) for x in sigmas_str.split(",")]
except:
    ALPHAS_S = [0.85, 0.92, 0.97]; SIGMAS_S = [0.3, 0.5, 0.8]

with st.spinner("📊 Menghitung sensitivitas parameter..."):
    SENS = run_sensitivity(X3_bytes, K_opt, ALPHAS_S, SIGMAS_S, T0, 1e-3, max_iter_sa)

# ──────────────────────────────────────────────────────────────────────────
#  EXECUTIVE METRIC STRIP
# ──────────────────────────────────────────────────────────────────────────
imp_wcss = (WCSS_BASE - RES["wcss_final"]) / WCSS_BASE * 100
imp_sil  = RES["sil"] - SIL_BASE

metric_cards([
    ("Pelanggan", f"{len(df)}", None, ""),
    ("Cluster Optimal (K)", f"{K_opt}", f"Silhouette {df_k.loc[K_opt,'Silhouette']:.4f}", ACCENT),
    ("SA WCSS Final", f"{RES['wcss_final']:.2f}", f"{'↓ ' if imp_wcss>0 else '↑ '}{abs(imp_wcss):.1f}% vs K-Means++",
     "#26A69A" if imp_wcss>0 else "#EF5350"),
    ("Silhouette SA", f"{RES['sil']:.4f}", f"{'↑' if imp_sil>=0 else '↓'} {abs(imp_sil):.4f} vs baseline",
     "#26A69A" if imp_sil>=0 else "#EF5350"),
    ("SA Iterasi", f"{RES['iters']}", f"Waktu {RES['time']:.2f}s", MUTED),
    ("Acceptance Rate", f"{RES['accept_rate']:.1f}%", f"Metro {RES['n_metro']} | Escape {RES['n_escape']}", MUTED),
])

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
#  TABS
# ──────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 EDA",
    "🔵 PCA",
    "🎯 Elbow & K",
    "🔥 Simulated Annealing",
    "🗺 Clustering",
    "👤 Profil Cluster",
    "⚗️ Sensitivitas",
    "⚖️ Perbandingan",
])

# ╔══════════════════════════════════════════════════════════════════════════
#  TAB 1 — EDA
# ╚══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("📊", "Eksplorasi Data Awal (EDA)")

    c1, c2 = st.columns([2, 1])
    with c1:
        # Histogram grid
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Age","Annual Income (k$)","Spending Score"])
        for i, col in enumerate(NUM_COLS):
            fig.add_trace(go.Histogram(
                x=df[col], name=col, nbinsx=20,
                marker_color=PAL[i], opacity=0.85,
                showlegend=False,
            ), row=1, col=i+1)
            mean_v = df[col].mean()
            fig.add_vline(x=mean_v, line_dash="dash", line_color="white",
                          line_width=1.5, row=1, col=i+1)
        fig.update_layout(**PLOTLY_LAYOUT, title="Distribusi Fitur Numerik", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        gc = df["Genre"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=gc.index, values=gc.values,
            hole=0.55,
            marker_colors=["#AB47BC","#42A5F5"],
            textfont_color=FG,
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Distribusi Genre",
                           height=320, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Scatter matrix
        pairs = [("Age","Spending Score (1-100)"),
                 ("Annual Income (k$)","Spending Score (1-100)"),
                 ("Age","Annual Income (k$)")]
        fig3 = make_subplots(rows=1, cols=3,
                             subplot_titles=["Age vs Score","Income vs Score","Age vs Income"])
        for i, (cx, cy) in enumerate(pairs):
            fig3.add_trace(go.Scatter(
                x=df[cx], y=df[cy], mode="markers",
                marker=dict(color=PAL[i], size=5, opacity=0.5),
                showlegend=False,
            ), row=1, col=i+1)
        fig3.update_layout(**PLOTLY_LAYOUT, title="Scatter Plots", height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        corr = df[NUM_COLS].corr()
        fig4 = go.Figure(go.Heatmap(
            z=corr.values,
            x=["Age","Income","Score"],
            y=["Age","Income","Score"],
            colorscale=[[0,"#EF5350"],[0.5,CARD],[1,"#26A69A"]],
            zmin=-1, zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            showscale=True,
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, title="Matriks Korelasi", height=320)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**📋 Statistik Deskriptif**")
    st.dataframe(df[NUM_COLS].describe().round(2)
                 .style.background_gradient(cmap="Blues", axis=1),
                 use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════
#  TAB 2 — PCA
# ╚══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("🔵", "Principal Component Analysis (PCA)")

    feat_short = ["Genre","Age","Income","Score"]
    load_df = pd.DataFrame(pca3.components_.T, index=FEATURES,
                           columns=[f"PC{i+1}" for i in range(n_pca)])

    c1, c2 = st.columns([1, 1])
    with c1:
        # Scree plot
        xp = list(range(1, len(EV)+1))
        colors_bar = [PAL[0] if i < n_pca else "#3D3F52" for i in range(len(EV))]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=xp, y=(EV*100).tolist(),
            marker_color=colors_bar, name="Explained Variance",
            text=[f"{v:.1f}%" for v in EV*100],
            textposition="outside",
        ))
        fig.add_trace(go.Scatter(
            x=xp, y=(EV_CUM*100).tolist(),
            mode="lines+markers", name="Kumulatif",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=7, color=ACCENT),
        ))
        fig.add_hline(y=80, line_dash="dot", line_color="white",
                      line_width=1, annotation_text="80% Threshold",
                      annotation_font_color=MUTED)
        fig.add_vline(x=n_pca+0.5, line_dash="dash", line_color=PAL[1], line_width=1.5)
        fig.update_layout(**PLOTLY_LAYOUT, title=f"Scree Plot  — PC1–PC{n_pca} dipilih ({EV_CUM[n_pca-1]*100:.1f}% var)",
                          height=380, yaxis_title="Explained Variance (%)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Biplot (PC1 vs PC2)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=X2[:,0], y=X2[:,1], mode="markers",
            marker=dict(color="#3D3F52", size=5, opacity=0.35),
            name="Data Points", showlegend=False,
        ))
        for i, feat in enumerate(FEATURES):
            scale = 3.0
            lx, ly = pca2.components_[0,i]*scale, pca2.components_[1,i]*scale
            fig2.add_annotation(x=lx, y=ly, ax=0, ay=0,
                                xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True,
                                arrowhead=2, arrowwidth=2.5, arrowcolor=PAL[i])
            fig2.add_annotation(x=lx*1.2, y=ly*1.2, text=feat_short[i],
                                font=dict(color=PAL[i], size=12), showarrow=False)
        fig2.update_layout(**PLOTLY_LAYOUT, title="Biplot — Loading Vektor",
                           height=380, xaxis_title=f"PC1 ({EV[0]*100:.1f}%)",
                           yaxis_title=f"PC2 ({EV[1]*100:.1f}%)")
        st.plotly_chart(fig2, use_container_width=True)

    # Loading heatmap
    lv = pca3.components_
    fig3 = go.Figure(go.Heatmap(
        z=lv, x=feat_short,
        y=[f"PC{i+1}" for i in range(n_pca)],
        colorscale=[[0,"#EF5350"],[0.5,CARD],[1,"#5C6BC0"]],
        zmin=-1, zmax=1,
        text=np.round(lv,3),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig3.update_layout(**PLOTLY_LAYOUT, title="Heatmap Loading Matrix", height=280)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 Detail Loading Matrix"):
        st.dataframe(load_df.round(4), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════
#  TAB 3 — ELBOW & K
# ╚══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("🎯", f"Penentuan K Optimal  (K = {K_opt})")

    kl = list(df_k.index)
    metrics_info = [
        ("WCSS", df_k["WCSS"].tolist(), True, PAL[0]),
        ("Silhouette", df_k["Silhouette"].tolist(), False, PAL[1]),
        ("Davies-Bouldin", df_k["Davies_Bouldin"].tolist(), True, PAL[2]),
        ("Calinski-H", df_k["Calinski_H"].tolist(), False, PAL[3]),
    ]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"{m} ({'↓' if l else '↑'} terbaik)"
                                        for m, _, l, _ in metrics_info])
    for idx, (name, vals, low, col) in enumerate(metrics_info):
        r, c = divmod(idx, 2)
        best_k_idx = np.argmin(vals) if low else np.argmax(vals)
        fig.add_trace(go.Scatter(
            x=kl, y=vals, mode="lines+markers",
            line=dict(color=col, width=2.5),
            marker=dict(size=8, color=col,
                        symbol=["circle"]*len(kl)),
            name=name,
        ), row=r+1, col=c+1)
        fig.add_vline(x=kl[best_k_idx], line_dash="dot", line_color="white",
                      line_width=1.5, row=r+1, col=c+1)
        fig.add_trace(go.Scatter(
            x=[kl[best_k_idx]], y=[vals[best_k_idx]],
            mode="markers",
            marker=dict(size=14, color=col,
                        symbol="star", line=dict(color="white", width=1.5)),
            showlegend=False, name=f"Optimal K={kl[best_k_idx]}",
        ), row=r+1, col=c+1)

    fig.update_layout(**PLOTLY_LAYOUT, title=f"Pemilihan K Optimal (k_max={k_max})", height=600)
    fig.update_xaxes(tickvals=kl, gridcolor=GRID_C)
    fig.update_yaxes(gridcolor=GRID_C)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_k.style.highlight_max(subset=["Silhouette","Calinski_H"], color="#1e3a2f")
                           .highlight_min(subset=["WCSS","Davies_Bouldin"], color="#1e3a2f"),
                 use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════
#  TAB 4 — SIMULATED ANNEALING
# ╚══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("🔥", "Mekanisme Simulated Annealing")

    hist_df = RES["hist"]
    iters_x = hist_df["iter"].values

    c1, c2 = st.columns(2)
    with c1:
        # WCSS convergence
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iters_x, y=hist_df["wcss_curr"].values,
            mode="lines", name="WCSS Saat Ini",
            line=dict(color=PAL[0], width=0.9), opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=iters_x, y=hist_df["wcss_best"].values,
            mode="lines", name="WCSS Terbaik (SA)",
            line=dict(color=ACCENT, width=2.5),
        ))
        fig.add_hline(y=WCSS_BASE, line_dash="dash", line_color=PAL[2],
                      annotation_text=f"K-Means++ baseline ({WCSS_BASE:.2f})",
                      annotation_font_color=PAL[2])
        fig.add_hline(y=RES["wcss_final"], line_dash="dot", line_color=PAL[1],
                      annotation_text=f"SA Final ({RES['wcss_final']:.2f})",
                      annotation_font_color=PAL[1])
        fig.update_layout(**PLOTLY_LAYOUT, title="Konvergensi WCSS",
                          xaxis_title="Iterasi", yaxis_title="WCSS", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Cooling schedule
        fig2 = go.Figure()
        t_vals = RES["t_log"]
        n = len(t_vals)
        fig2.add_trace(go.Scatter(
            x=list(range(n)), y=t_vals,
            mode="lines", name="Suhu T",
            line=dict(color=PAL[2], width=2),
            fill="tozeroy", fillcolor="rgba(239,83,80,.08)",
        ))
        for zone, x0, x1, col, label in [
            ("Eksplorasi", 0, int(n*0.20), PAL[2], "Eksplorasi"),
            ("Transisi",   int(n*0.20), int(n*0.65), ACCENT, "Transisi"),
            ("Eksploitasi", int(n*0.65), n-1, PAL[1], "Eksploitasi"),
        ]:
            fig2.add_vrect(x0=x0, x1=x1, fillcolor=col, opacity=0.06,
                           annotation_text=label, annotation_position="top left",
                           annotation_font_color=col)
        fig2.update_yaxes(type="log")
        fig2.update_layout(**PLOTLY_LAYOUT, title=f"Cooling Schedule  α={alpha}",
                           xaxis_title="Iterasi", yaxis_title="Suhu T (log)", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Delta E distribution
        d_neg = hist_df["delta"][hist_df["delta"] < 0]
        d_pos = hist_df["delta"][hist_df["delta"] > 0]
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=d_neg, name=f"ΔE<0 Accept ({len(d_neg)})",
            marker_color=PAL[1], opacity=0.85, nbinsx=35,
        ))
        fig3.add_trace(go.Histogram(
            x=d_pos, name=f"ΔE>0 Metropolis ({len(d_pos)})",
            marker_color=PAL[2], opacity=0.85, nbinsx=35,
        ))
        fig3.add_vline(x=0, line_dash="dash", line_color="white")
        fig3.update_layout(**PLOTLY_LAYOUT, title="Distribusi ΔE per Iterasi",
                           barmode="overlay", xaxis_title="ΔE", yaxis_title="Frekuensi",
                           height=380)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Acceptance rate per block
        block = max(20, len(hist_df)//30)
        acc_blk, centers_blk = [], []
        for i in range(0, len(hist_df), block):
            chunk = hist_df.iloc[i:i+block]
            n_acc = (chunk["status"] == "Accept").sum()
            acc_blk.append(n_acc / len(chunk) * 100)
            centers_blk.append(i + block//2)

        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Bar(
            x=centers_blk, y=acc_blk,
            name="Acceptance Rate (%)",
            marker_color=PAL[0], opacity=0.8,
        ), secondary_y=False)
        fig4.add_trace(go.Scatter(
            x=list(range(len(RES["t_log"]))), y=RES["t_log"],
            mode="lines", name="Suhu T",
            line=dict(color=PAL[2], width=1.5, dash="dot"),
        ), secondary_y=True)
        fig4.add_hline(y=50, line_dash="dot", line_color="white", line_width=1, secondary_y=False)
        fig4.update_yaxes(type="log", secondary_y=True, title_text="Suhu T", gridcolor=GRID_C)
        fig4.update_layout(**PLOTLY_LAYOUT, title="Acceptance Rate per Blok + Cooling",
                           xaxis_title="Iterasi", yaxis_title="Acceptance Rate (%)", height=380)
        st.plotly_chart(fig4, use_container_width=True)

    # Metropolis curve
    section_header("📐", "Kurva Probabilitas Metropolis (Teoritis)")
    T_axis = np.linspace(0.01, T0, 500)
    fig5 = go.Figure()
    for de, col_m in zip([0.5, 1.0, 2.0, 5.0, 10.0], PAL):
        probs = np.exp(-de / T_axis)
        fig5.add_trace(go.Scatter(
            x=T_axis, y=probs, mode="lines",
            name=f"ΔE={de:.1f}",
            line=dict(color=col_m, width=2),
        ))
    fig5.update_layout(**PLOTLY_LAYOUT, title="P(terima) = exp(−ΔE/T)",
                       xaxis_title="Suhu T", yaxis_title="P(terima)", height=360)
    st.plotly_chart(fig5, use_container_width=True)

    # SA Summary stats
    with st.expander("📊 Ringkasan SA"):
        total_sa = RES["n_better"] + RES["n_metro"] + RES["n_reject"]
        scols = st.columns(4)
        for col, (k_lbl, v) in zip(scols, [
            ("Accept ΔE<0", f"{RES['n_better']} ({RES['n_better']/total_sa*100:.1f}%)"),
            ("Accept Metro", f"{RES['n_metro']} ({RES['n_metro']/total_sa*100:.1f}%)"),
            ("Reject",       f"{RES['n_reject']} ({RES['n_reject']/total_sa*100:.1f}%)"),
            ("Escape Local", str(RES["n_escape"])),
        ]):
            col.metric(k_lbl, v)

    with st.expander("📋 Tabel Iterasi (50 baris)"):
        st.dataframe(hist_df.head(50), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════
#  TAB 5 — CLUSTERING (2D + 3D)
# ╚══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("🗺", f"Hasil Clustering SA+K-Means  (K={K_opt})")

    c1, c2 = st.columns([1, 1])
    with c1:
        # 2D PCA scatter
        cents_2d = pca2.transform(pca3.inverse_transform(RES["centers"]))
        fig = go.Figure()
        for c in cluster_ids:
            m = RES["labels"] == c
            fig.add_trace(go.Scatter(
                x=X2[m,0], y=X2[m,1], mode="markers",
                name=f"Cluster {c}  (n={m.sum()})",
                marker=dict(color=PAL[c%len(PAL)], size=7, opacity=0.75,
                            line=dict(color="white", width=0.4)),
            ))
        fig.add_trace(go.Scatter(
            x=cents_2d[:,0], y=cents_2d[:,1], mode="markers",
            name="Centroid", marker=dict(color="white", size=16,
                                          symbol="star", line=dict(color="black", width=1)),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Proyeksi 2D (PC1 vs PC2)",
                          xaxis_title=f"PC1 ({EV[0]*100:.1f}%)",
                          yaxis_title=f"PC2 ({EV[1]*100:.1f}%)", height=460)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # 3D PCA scatter
        fig3d = go.Figure()
        for c in cluster_ids:
            m = RES["labels"] == c
            fig3d.add_trace(go.Scatter3d(
                x=X3[m,0], y=X3[m,1], z=X3[m,2],
                mode="markers", name=f"Cluster {c}",
                marker=dict(color=PAL[c%len(PAL)], size=4, opacity=0.65),
            ))
        fig3d.add_trace(go.Scatter3d(
            x=RES["centers"][:,0], y=RES["centers"][:,1],
            z=RES["centers"][:,2] if n_pca >= 3 else np.zeros(K_opt),
            mode="markers", name="Centroid",
            marker=dict(color="white", size=8, symbol="diamond"),
        ))
        fig3d.update_layout(
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            font=dict(color=FG),
            title=dict(text="Proyeksi 3D (PC1–PC3)", font=dict(size=14)),
            scene=dict(
                bgcolor=CARD,
                xaxis=dict(title="PC1", gridcolor=GRID_C, backgroundcolor=CARD),
                yaxis=dict(title="PC2", gridcolor=GRID_C, backgroundcolor=CARD),
                zaxis=dict(title="PC3", gridcolor=GRID_C, backgroundcolor=CARD),
            ),
            height=460,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # Original feature scatter colored by cluster
    section_header("🎨", "Scatter Fitur Asli per Cluster")
    c3, c4, c5 = st.columns(3)
    pairs_orig = [
        ("Age","Spending Score (1-100)","Age vs Spending Score"),
        ("Annual Income (k$)","Spending Score (1-100)","Income vs Spending Score"),
        ("Age","Annual Income (k$)","Age vs Annual Income"),
    ]
    for col_st, (cx, cy, title_p) in zip([c3,c4,c5], pairs_orig):
        fig_p = go.Figure()
        for c in cluster_ids:
            m = df["Cluster"] == c
            fig_p.add_trace(go.Scatter(
                x=df.loc[m, cx], y=df.loc[m, cy], mode="markers",
                name=f"C{c}", marker=dict(color=PAL[c%len(PAL)], size=7, opacity=0.7),
            ))
        fig_p.update_layout(**PLOTLY_LAYOUT, title=title_p,
                            xaxis_title=cx, yaxis_title=cy, height=340,
                            showlegend=True)
        col_st.plotly_chart(fig_p, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════
#  TAB 6 — PROFIL CLUSTER
# ╚══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section_header("👤", "Profil & Interpretasi Segmen Pelanggan")

    c1, c2, c3 = st.columns(3)

    with c1:
        # Bar grouped means
        fig = go.Figure()
        for i, col in enumerate(NUM_COLS):
            fig.add_trace(go.Bar(
                x=[f"C{c}" for c in cluster_ids],
                y=[profile.loc[c, col] for c in cluster_ids],
                name=col.split("(")[0].strip(),
                marker_color=PAL[i], opacity=0.85,
            ))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group",
                          title="Rata-rata Fitur per Cluster", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Pie cluster size
        fig2 = go.Figure(go.Pie(
            labels=[f"Cluster {c}" for c in cluster_ids],
            values=[profile.loc[c,"N"] for c in cluster_ids],
            hole=0.5,
            marker_colors=PAL[:K_opt],
            textfont_color=FG,
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Distribusi Anggota", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        # Normalized heatmap
        hd = profile[NUM_COLS].values.astype(float)
        hn = (hd - hd.min(0)) / (hd.max(0) - hd.min(0) + 1e-9)
        fig3 = go.Figure(go.Heatmap(
            z=hn,
            x=["Age","Income","Score"],
            y=[f"C{c}" for c in cluster_ids],
            colorscale=[[0,CARD],[0.5,PAL[0]],[1,PAL[1]]],
            text=np.round(hd, 1),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, title="Heatmap Profil (Normalized)", height=380)
        st.plotly_chart(fig3, use_container_width=True)

    # Violin plots
    section_header("🎻", "Distribusi Fitur per Cluster (Violin)")
    cv1, cv2, cv3 = st.columns(3)
    for col_st, feat in zip([cv1, cv2, cv3], NUM_COLS):
        fig_v = go.Figure()
        for c in cluster_ids:
            data_c = df[df["Cluster"]==c][feat].values
            fig_v.add_trace(go.Violin(
                y=data_c, name=f"C{c}",
                box_visible=True, meanline_visible=True,
                fillcolor=PAL[c%len(PAL)], opacity=0.75,
                line_color="white",
            ))
        fig_v.update_layout(**PLOTLY_LAYOUT,
                            title=feat.split("(")[0].strip(),
                            showlegend=False, height=340)
        col_st.plotly_chart(fig_v, use_container_width=True)

    # Radar charts
    section_header("🕸", "Radar Chart Profil Cluster")
    rd = profile[NUM_COLS].copy()
    for col in rd.columns:
        rd[col] = (rd[col]-rd[col].min())/(rd[col].max()-rd[col].min()+1e-9)
    cats = ["Age","Annual Income","Spending Score"]

    radar_cols = st.columns(K_opt)
    for ci, (c, col_st) in enumerate(zip(cluster_ids, radar_cols)):
        vals = rd.loc[c].tolist()
        vals_closed = vals + [vals[0]]
        cats_closed  = cats + [cats[0]]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", fillcolor=PAL[ci%len(PAL)],
            line_color=PAL[ci%len(PAL)],
            opacity=0.65, name=f"Cluster {c}",
        ))
        fig_r.update_layout(
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            font=dict(color=FG),
            polar=dict(
                bgcolor=CARD,
                radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID_C, color=MUTED),
                angularaxis=dict(gridcolor=GRID_C, color=FG),
            ),
            title=dict(text=f"Cluster {c}  (n={int(profile.loc[c,'N'])})", font=dict(size=13)),
            height=320, margin=dict(l=30, r=30, t=60, b=30),
            showlegend=False,
        )
        col_st.plotly_chart(fig_r, use_container_width=True)

    # Segment interpretation table
    section_header("📝", "Interpretasi Segmen")
    rows_interp = []
    for c in cluster_ids:
        r = profile.loc[c]
        inc_lbl = "Tinggi" if r["Annual Income (k$)"] > 65 else ("Sedang" if r["Annual Income (k$)"] > 42 else "Rendah")
        sco_lbl = "Tinggi" if r["Spending Score (1-100)"] > 60 else ("Sedang" if r["Spending Score (1-100)"] > 40 else "Rendah")
        rows_interp.append({
            "Cluster": f"C{c}", "Usia Rata-rata": f"{r['Age']:.1f}",
            "Income Rata-rata": f"{r['Annual Income (k$)']:.1f}k$",
            "Score Rata-rata": f"{r['Spending Score (1-100)']:.1f}",
            "Anggota": f"{int(r['N'])} ({r['%']:.1f}%)",
            "Profil": f"Income {inc_lbl} · Spending {sco_lbl}",
        })
    st.dataframe(pd.DataFrame(rows_interp), use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════
#  TAB 7 — SENSITIVITAS
# ╚══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    section_header("⚗️", f"Analisis Sensitivitas Parameter SA  (α×σ grid)")

    # Heatmap WCSS
    heat_wcss = np.array([[r["wcss_final"] for r in SENS if r["alpha"] == a]
                           for a in ALPHAS_S])
    heat_sil  = np.array([[r["sil"] for r in SENS if r["alpha"] == a]
                           for a in ALPHAS_S])

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Heatmap(
            z=heat_wcss,
            x=[f"σ={s}" for s in SIGMAS_S],
            y=[f"α={a}" for a in ALPHAS_S],
            colorscale=[[0,PAL[1]],[0.5,CARD],[1,PAL[2]]],
            text=np.round(heat_wcss,2),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Heatmap WCSS Final: α × σ", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure(go.Heatmap(
            z=heat_sil,
            x=[f"σ={s}" for s in SIGMAS_S],
            y=[f"α={a}" for a in ALPHAS_S],
            colorscale=[[0,CARD],[0.5,PAL[0]],[1,PAL[1]]],
            text=np.round(heat_sil,4),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Heatmap Silhouette: α × σ (↑ terbaik)", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # Bar WCSS all combinations
    labels_b = [f"α={r['alpha']} σ={r['sigma']}" for r in SENS]
    wcss_b   = [r["wcss_final"] for r in SENS]
    col_b    = [PAL[1] if w < WCSS_BASE else PAL[2] for w in wcss_b]
    fig3 = go.Figure(go.Bar(
        x=labels_b, y=wcss_b,
        marker_color=col_b, opacity=0.85,
        text=[f"{w:.2f}" for w in wcss_b],
        textposition="outside",
    ))
    fig3.add_hline(y=WCSS_BASE, line_dash="dash", line_color="white", line_width=2,
                   annotation_text=f"K-Means++ Baseline ({WCSS_BASE:.2f})",
                   annotation_font_color=MUTED)
    fig3.update_layout(**PLOTLY_LAYOUT,
                       title="WCSS Final per Kombinasi  (biru = SA menang)",
                       yaxis_title="WCSS Final", height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # Convergence per alpha (sigma fixed)
    sigma_fixed = SIGMAS_S[len(SIGMAS_S)//2] if len(SIGMAS_S) > 1 else SIGMAS_S[0]
    section_header("📈", f"Konvergensi WCSS per α  (σ={sigma_fixed} tetap)")
    fig4 = go.Figure()
    for i, a in enumerate(ALPHAS_S):
        r_a = next((r for r in SENS if r["alpha"]==a and r["sigma"]==sigma_fixed), None)
        if r_a:
            fig4.add_trace(go.Scatter(
                y=r_a["wcss_best_log"], mode="lines",
                name=f"α={a}  final={r_a['wcss_final']:.2f}",
                line=dict(color=PAL[i], width=2),
            ))
    fig4.add_hline(y=WCSS_BASE, line_dash="dot", line_color="white",
                   annotation_text="Baseline")
    fig4.update_layout(**PLOTLY_LAYOUT, title=f"Konvergensi per α (σ={sigma_fixed})",
                       xaxis_title="Iterasi", yaxis_title="WCSS Terbaik", height=380)
    st.plotly_chart(fig4, use_container_width=True)

    # Escape count
    esc_vals = [r["n_escape"] for r in SENS]
    fig5 = go.Figure(go.Bar(
        x=labels_b, y=esc_vals,
        marker_color=[PAL[i//len(SIGMAS_S)%len(PAL)] for i in range(len(SENS))],
        text=esc_vals, textposition="outside",
    ))
    fig5.update_layout(**PLOTLY_LAYOUT, title="Kemampuan SA Keluar Local Minimum (Escape Count)",
                       yaxis_title="Estimasi Escape", height=360)
    st.plotly_chart(fig5, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════
#  TAB 8 — PERBANDINGAN
# ╚══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    section_header("⚖️", "Perbandingan Metode")

    km_r_sil = silhouette_score(X3, km_rand.labels_)
    km_r_db  = davies_bouldin_score(X3, km_rand.labels_)
    km_r_ch  = calinski_harabasz_score(X3, km_rand.labels_)

    methods_data = {
        "Metode": ["K-Means Random", "K-Means++", "SA + K-Means (Hybrid)"],
        "WCSS": [km_rand.inertia_, WCSS_BASE, RES["wcss_final"]],
        "Silhouette": [km_r_sil, SIL_BASE, RES["sil"]],
        "Davies-Bouldin": [km_r_db, DB_BASE, RES["db"]],
        "Calinski-H": [km_r_ch, CH_BASE, RES["ch"]],
    }
    df_comp = pd.DataFrame(methods_data)

    c1, c2, c3 = st.columns(3)
    for col_st, (metric, low, ytitle) in zip([c1,c2,c3], [
        ("WCSS", True, "WCSS (↓ lebih baik)"),
        ("Silhouette", False, "Silhouette (↑ lebih baik)"),
        ("Davies-Bouldin", True, "DB Index (↓ lebih baik)"),
    ]):
        vals = df_comp[metric].tolist()
        best_i = np.argmin(vals) if low else np.argmax(vals)
        colors_m = [PAL[3], PAL[0], PAL[1]]
        edges    = ["rgba(0,0,0,0)"]*3
        edges[best_i] = "white"
        widths = [0]*3; widths[best_i] = 2
        fig = go.Figure(go.Bar(
            x=df_comp["Metode"], y=vals,
            marker_color=colors_m,
            marker_line_color=edges,
            marker_line_width=widths,
            opacity=0.85,
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title=ytitle, yaxis_title=metric, height=400)
        col_st.plotly_chart(fig, use_container_width=True)

    # Delta table
    imp_wcss_v  = (WCSS_BASE - RES["wcss_final"]) / WCSS_BASE * 100
    imp_sil_v   = (RES["sil"] - SIL_BASE) / SIL_BASE * 100
    imp_db_v    = (DB_BASE - RES["db"]) / DB_BASE * 100
    imp_ch_v    = (RES["ch"] - CH_BASE) / CH_BASE * 100

    section_header("📊", "Delta SA+K-Means vs K-Means++")
    delta_cols = st.columns(4)
    for col_st, (label, delta_v, improve_dir) in zip(delta_cols, [
        ("WCSS Δ%", imp_wcss_v, "pos_good"),
        ("Silhouette Δ%", imp_sil_v, "pos_good"),
        ("DB Index Δ%", imp_db_v, "pos_good"),
        ("Calinski-H Δ%", imp_ch_v, "pos_good"),
    ]):
        color = "#26A69A" if delta_v > 0 else "#EF5350"
        col_st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color}">{delta_v:+.2f}%</div>
          <div class="metric-delta" style="color:{color}">
            {'✓ Membaik' if delta_v > 0 else '✗ Memburuk'}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df_comp.set_index("Metode").style
                 .highlight_min(subset=["WCSS","Davies-Bouldin"], color="#1e3a2f")
                 .highlight_max(subset=["Silhouette","Calinski-H"], color="#1e3a2f"),
                 use_container_width=True)

    # Radar comparison
    section_header("🕸", "Radar Perbandingan Multi-Metrik")
    norm_metrics = {}
    for m in ["WCSS","Silhouette","Davies-Bouldin","Calinski-H"]:
        col_vals = df_comp[m].values
        mn, mx = col_vals.min(), col_vals.max()
        norm_metrics[m] = (col_vals - mn) / (mx - mn + 1e-9)

    cats_r = ["WCSS (↓)","Silhouette (↑)","Davies-Bouldin (↓)","Calinski-H (↑)"]
    fig_r = go.Figure()
    for i, method in enumerate(df_comp["Metode"]):
        vals_r = [
            1 - norm_metrics["WCSS"][i],
            norm_metrics["Silhouette"][i],
            1 - norm_metrics["Davies-Bouldin"][i],
            norm_metrics["Calinski-H"][i],
        ]
        vals_r_closed = vals_r + [vals_r[0]]
        cats_closed_r  = cats_r + [cats_r[0]]
        fig_r.add_trace(go.Scatterpolar(
            r=vals_r_closed, theta=cats_closed_r,
            fill="toself", name=method,
            line_color=PAL[i%len(PAL)],
            fillcolor=PAL[i%len(PAL)],
            opacity=0.4,
        ))
    fig_r.update_layout(
        paper_bgcolor=CARD,
        font=dict(color=FG),
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID_C, color=MUTED),
            angularaxis=dict(gridcolor=GRID_C, color=FG),
        ),
        title="Radar Multi-Metrik (dinormalisasi, semua ↑ = lebih baik)",
        height=500,
    )
    st.plotly_chart(fig_r, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#2C2F3E;margin:32px 0 16px'>
<div style='text-align:center;color:#7986CB;font-size:12px;padding-bottom:16px'>
  Mall Customer Segmentation Dashboard  ·  PCA + Simulated Annealing + K-Means  ·  Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
