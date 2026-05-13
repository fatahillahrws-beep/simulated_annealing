"""
Mall Customer Segmentation Dashboard
PCA + Simulated Annealing + K-Means Clustering
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math, random, time, itertools, warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
warnings.filterwarnings('ignore')

# ── KONFIGURASI HALAMAN ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS PROFESIONAL ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #0A0C14; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1117 0%, #131825 100%);
        border-right: 1px solid #1E2235;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    
    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 50%, #161924 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #5C6BC0, #26A69A, #FFA726, #EF5350);
    }
    .main-header h1 {
        font-size: 1.9rem; font-weight: 700;
        color: #E8EAF6; margin: 0 0 0.4rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #7986CB; margin: 0; font-size: 0.92rem; font-weight: 400;
    }
    
    /* Metric Card */
    .metric-card {
        background: #131825;
        border: 1px solid #1E2235;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
        height: 100%;
    }
    .metric-card:hover { border-color: #3D4F6B; }
    .metric-label {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em;
        color: #5C6A8A; text-transform: uppercase; margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.75rem; font-weight: 700;
        color: #E8EAF6; line-height: 1;
    }
    .metric-sub {
        font-size: 0.78rem; color: #4A5268; margin-top: 0.35rem;
    }
    .metric-badge {
        display: inline-block; font-size: 0.68rem; font-weight: 600;
        padding: 0.15rem 0.5rem; border-radius: 20px; margin-top: 0.4rem;
        letter-spacing: 0.04em;
    }
    .badge-green { background: #0d2e1a; color: #4ade80; border: 1px solid #166534; }
    .badge-blue  { background: #0d1e3a; color: #60a5fa; border: 1px solid #1e3a5f; }
    .badge-orange{ background: #2e1a0d; color: #fb923c; border: 1px solid #6b3010; }
    
    /* Section header */
    .section-header {
        display: flex; align-items: center; gap: 0.75rem;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #1E2235;
    }
    .section-dot {
        width: 4px; height: 22px; border-radius: 2px;
        background: linear-gradient(180deg, #5C6BC0, #26A69A);
        flex-shrink: 0;
    }
    .section-title {
        font-size: 1.05rem; font-weight: 600;
        color: #C5CBE0; letter-spacing: -0.2px;
    }
    .section-sub {
        font-size: 0.8rem; color: #4A5268; margin-left: auto;
    }
    
    /* Table */
    .dataframe { background: #0F1117 !important; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #0F1117;
        border-bottom: 1px solid #1E2235;
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #5C6A8A; font-size: 0.85rem; font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #A5B4FC !important;
        background: #131825 !important;
        border-bottom: 2px solid #5C6BC0 !important;
    }
    
    /* Divider */
    hr { border-color: #1E2235 !important; margin: 1rem 0 !important; }
    
    /* Slider & selectbox accent */
    .stSlider [data-testid="stThumbValue"] { color: #A5B4FC; }
    
    /* Info / warning boxes */
    .info-box {
        background: #0d1e3a; border-left: 3px solid #3B82F6;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        font-size: 0.83rem; color: #93C5FD; margin: 0.5rem 0;
    }
    .success-box {
        background: #0d2e1a; border-left: 3px solid #22C55E;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        font-size: 0.83rem; color: #86EFAC; margin: 0.5rem 0;
    }
    .warn-box {
        background: #2e1a0d; border-left: 3px solid #F97316;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        font-size: 0.83rem; color: #FDBA74; margin: 0.5rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }
    /* Preserve sidebar toggle button */
    [data-testid="stHeader"] button,
    [data-testid="collapsedControl"],
    button[kind="header"],
    button[title="Open sidebar"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"] {
        visibility: visible !important;
        pointer-events: auto !important;
    }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── PALETTE WARNA ─────────────────────────────────────────────────────────────
PAL = ['#5C6BC0','#26A69A','#EF5350','#FFA726','#66BB6A','#AB47BC','#EC407A']
PLOTLY_THEME = dict(
    plot_bgcolor='#131825',
    paper_bgcolor='#0A0C14',
    font=dict(color='#C5CBE0', family='Inter'),
    xaxis=dict(gridcolor='#1E2235', linecolor='#1E2235', zerolinecolor='#1E2235'),
    yaxis=dict(gridcolor='#1E2235', linecolor='#1E2235', zerolinecolor='#1E2235'),
    colorway=PAL,
    margin=dict(l=50, r=30, t=50, b=50),
)

# Untuk make_subplots — tidak include xaxis/yaxis/colorway/margin
SUBPLOT_THEME = dict(
    plot_bgcolor='#131825',
    paper_bgcolor='#0A0C14',
    font=dict(color='#C5CBE0', family='Inter'),
)

def apply_theme(fig, title=""):
    fig.update_layout(**PLOTLY_THEME, title=dict(text=title, font=dict(size=13, color='#C5CBE0'), x=0.01))
    return fig

# ── UPLOAD DATA ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes, sep=';'):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    return df

@st.cache_data
def preprocess_data(df):
    le = LabelEncoder()
    df = df.copy()
    df['Genre_enc'] = le.fit_transform(df['Genre'])
    FEATURES = ['Genre_enc','Age','Annual Income (k$)','Spending Score (1-100)']
    X_raw = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return df, X_scaled, scaler, FEATURES

@st.cache_data
def run_pca(X_scaled, n_clu=3, n_viz=2):
    pca_full = PCA().fit(X_scaled)
    EV = pca_full.explained_variance_ratio_
    EV_CUM = np.cumsum(EV)
    pca3 = PCA(n_components=n_clu).fit(X_scaled)
    pca2 = PCA(n_components=n_viz).fit(X_scaled)
    X3 = pca3.transform(X_scaled)
    X2 = pca2.transform(X_scaled)
    return pca_full, pca3, pca2, X3, X2, EV, EV_CUM

@st.cache_data
def find_optimal_k(X3):
    K_RANGE = range(2, 11)
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42).fit(X3)
        lbl = km.labels_
        rows.append({'K': k, 'WCSS': km.inertia_,
                     'Silhouette': silhouette_score(X3, lbl),
                     'Davies-Bouldin': davies_bouldin_score(X3, lbl),
                     'Calinski-H': calinski_harabasz_score(X3, lbl)})
    df_k = pd.DataFrame(rows).set_index('K')
    K = int(df_k['Silhouette'].idxmax())
    return df_k, K

def run_sa_cached(X, K, T0=100.0, alpha=0.95, T_min=1e-3, max_iter=600, sigma=0.5, seed=42):
    np.random.seed(seed); random.seed(seed)
    WCSS_BASE = KMeans(n_clusters=K, init='k-means++', n_init=30, random_state=42).fit(X).inertia_

    def eval_wcss(centers):
        km = KMeans(n_clusters=K, init=centers, n_init=1, max_iter=300, random_state=seed)
        km.fit(X)
        return km.inertia_, km.labels_, km.cluster_centers_

    idx0 = np.random.choice(len(X), K, replace=False)
    c_curr = X[idx0].copy()
    wcss_curr, lbl_curr, c_curr = eval_wcss(c_curr)
    c_best = c_curr.copy()
    wcss_best = wcss_curr
    wcss_init = wcss_curr
    T = T0
    t_log = [T]; wcss_log = [wcss_curr]; wcss_best_log = [wcss_best]
    history = []
    n_accept_better = 0; n_accept_metro = 0; n_reject = 0; n_escape = 0
    prev_was_worse = False
    it = 0; t0 = time.time()

    while T > T_min and it < max_iter:
        it += 1
        c_cand = c_curr + np.random.normal(0, sigma, c_curr.shape)
        wcss_cand, lbl_cand, c_cand = eval_wcss(c_cand)
        delta = wcss_cand - wcss_curr
        if delta < 0:
            prob = 1.0; accept = True; status = 'Accept'
            n_accept_better += 1
        else:
            prob = math.exp(-delta / T)
            accept = (random.random() < prob)
            status = 'Accept*' if accept else 'Reject'
            if accept: n_accept_metro += 1
            else: n_reject += 1
        if accept:
            c_curr = c_cand.copy(); wcss_curr = wcss_cand
            if prev_was_worse and wcss_curr < wcss_best: n_escape += 1
            prev_was_worse = (delta >= 0)
        else:
            prev_was_worse = False
        if wcss_curr < wcss_best:
            wcss_best = wcss_curr; c_best = c_curr.copy()
        history.append({'iter': it, 'T': round(T,6), 'wcss_curr': round(wcss_curr,4),
                         'wcss_best': round(wcss_best,4), 'delta': round(delta,4),
                         'prob': round(prob,6), 'status': status})
        t_log.append(T); wcss_log.append(wcss_curr); wcss_best_log.append(wcss_best)
        T *= alpha

    elapsed = time.time() - t0
    wcss_final, lbl_final, c_final = eval_wcss(c_best)
    total = n_accept_better + n_accept_metro + n_reject
    return {
        'wcss_init': wcss_init, 'wcss_best_sa': wcss_best, 'wcss_final': wcss_final,
        'labels': lbl_final, 'centers': c_final, 'hist': pd.DataFrame(history),
        't_log': t_log, 'wcss_log': wcss_log, 'wcss_best_log': wcss_best_log,
        'iters': it, 'time': elapsed, 'accept_rate': (n_accept_better+n_accept_metro)/total*100,
        'n_metro': n_accept_metro, 'n_better': n_accept_better,
        'n_reject': n_reject, 'n_escape': n_escape,
        'sil': silhouette_score(X, lbl_final),
        'db': davies_bouldin_score(X, lbl_final),
        'ch': calinski_harabasz_score(X, lbl_final),
        'WCSS_BASE': WCSS_BASE
    }

# ── LOAD & PROSES DATA ────────────────────────────────────────────────────────
# Upload widget di luar sidebar agar muncul di atas saat belum ada file
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None

with st.sidebar:
    st.markdown("**Upload Dataset**")
    uploaded_file = st.file_uploader(
        "Upload file CSV", type=["csv"],
        help="Format: CustomerID, Genre, Age, Annual Income (k$), Spending Score (1-100)"
    )
    sep_choice = st.radio("Separator", [";", ","], horizontal=True)

if uploaded_file is not None:
    try:
        df_raw = load_data(uploaded_file.read(), sep=sep_choice)
        st.session_state.uploaded_df = df_raw
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()
elif st.session_state.uploaded_df is not None:
    df_raw = st.session_state.uploaded_df
else:
    st.markdown("""
    <div style='background:#131825; border:1px solid #2a2d3e; border-radius:12px;
                padding:3rem; text-align:center; margin-top:2rem;'>
        <div style='font-size:2.5rem; margin-bottom:1rem;'>📂</div>
        <div style='font-size:1.1rem; font-weight:600; color:#C5CBE0; margin-bottom:0.5rem;'>
            Upload Dataset untuk Memulai
        </div>
        <div style='font-size:0.85rem; color:#5C6A8A; margin-bottom:1.5rem;'>
            Upload file CSV Mall Customers melalui sidebar kiri.<br>
            Pastikan kolom: <b style='color:#7986CB;'>Genre, Age, Annual Income (k$), Spending Score (1-100)</b>
        </div>
        <div style='background:#0A0C14; border-radius:8px; padding:1rem; display:inline-block;
                    font-size:0.8rem; color:#4A5568; text-align:left;'>
            CustomerID;Genre;Age;Annual Income (k$);Spending Score (1-100)<br>
            1;Male;19;15;39<br>
            2;Male;21;15;81<br>
            3;Female;20;16;6<br>
            ...
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Validasi kolom
required_cols = {'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'}
missing_cols = required_cols - set(df_raw.columns)
if missing_cols:
    st.error(f"Kolom tidak ditemukan: {missing_cols}. Pastikan nama kolom sudah benar.")
    st.stop()

df, X_scaled, scaler, FEATURES = preprocess_data(df_raw)
pca_full, pca3, pca2, X3, X2, EV, EV_CUM = run_pca(X_scaled)
df_k, K_OPT = find_optimal_k(X3)
N_CLU = 3; N_VIZ = 2
feat_short = ['Genre','Age','Income','Score']
FEATURES_DISPLAY = ['Genre_enc','Age','Annual Income (k$)','Spending Score (1-100)']
NUM_COLS = ['Age','Annual Income (k$)','Spending Score (1-100)']

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.2rem; border-bottom: 1px solid #1E2235; margin-bottom: 1.2rem;'>
        <div style='font-size:0.7rem; color:#5C6A8A; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem;'>
            SEGMENTASI PELANGGAN
        </div>
        <div style='font-size:1.15rem; font-weight:700; color:#E8EAF6;'>
            SA + PCA + K-Means
        </div>
        <div style='font-size:0.78rem; color:#4A5268; margin-top:0.25rem;'>
            Mall Customer Dataset · 200 samples
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Parameter Simulated Annealing**")
    
    T0 = st.slider("Suhu Awal (T₀)", 10.0, 200.0, 100.0, 10.0)
    alpha = st.slider("Cooling Rate (α)", 0.80, 0.99, 0.95, 0.01)
    sigma = st.slider("Perturbasi (σ)", 0.1, 1.5, 0.5, 0.1)
    max_iter = st.slider("Max Iterasi", 100, 1000, 600, 50)
    K_input = st.slider("Jumlah Cluster (K)", 2, 10, K_OPT, 1)
    
    st.markdown("---")
    run_btn = st.button("▶  Jalankan SA", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#3A4260; line-height:1.7;'>
        <b style='color:#4A5568;'>Panduan Parameter</b><br>
        • <b style='color:#5C6BC0;'>α rendah</b> = konvergen cepat<br>
        • <b style='color:#26A69A;'>α tinggi</b> = eksplorasi luas<br>
        • <b style='color:#FFA726;'>σ kecil</b> = gerak presisi<br>
        • <b style='color:#EF5350;'>σ besar</b> = berani keluar lokal
    </div>
    """, unsafe_allow_html=True)

# ── HEADER UTAMA ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏬 Mall Customer Segmentation</h1>
    <p>Analisis klasterisasi menggunakan PCA · Simulated Annealing · K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)

# ── JALANKAN SA ───────────────────────────────────────────────────────────────
if 'sa_result' not in st.session_state:
    with st.spinner("Menginisialisasi analisis..."):
        st.session_state.sa_result = run_sa_cached(X3, K_input, T0=T0, alpha=alpha,
                                                     sigma=sigma, max_iter=max_iter)
        st.session_state.k_used = K_input

if run_btn:
    with st.spinner("Menjalankan Simulated Annealing..."):
        progress = st.progress(0)
        st.session_state.sa_result = run_sa_cached(X3, K_input, T0=T0, alpha=alpha,
                                                     sigma=sigma, max_iter=max_iter)
        st.session_state.k_used = K_input
        progress.progress(100)
    st.success(f"SA selesai dalam {st.session_state.sa_result['time']:.2f} detik")

RES = st.session_state.sa_result
K = st.session_state.k_used if 'k_used' in st.session_state else K_OPT
WCSS_BASE = RES['WCSS_BASE']
SIL_BASE = silhouette_score(X3, KMeans(n_clusters=K, init='k-means++', n_init=30, random_state=42).fit(X3).labels_)
DB_BASE = davies_bouldin_score(X3, KMeans(n_clusters=K, init='k-means++', n_init=30, random_state=42).fit(X3).labels_)

df['Cluster'] = RES['labels']
cluster_ids = sorted(df['Cluster'].unique())
profile = df.groupby('Cluster')[NUM_COLS].mean().round(2)
profile['N'] = df.groupby('Cluster').size()
profile['%'] = (profile['N'] / len(df) * 100).round(1)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
imp_wcss = (WCSS_BASE - RES['wcss_final']) / WCSS_BASE * 100
imp_sil  = (RES['sil'] - SIL_BASE) / SIL_BASE * 100

c1, c2, c3, c4, c5 = st.columns(5)
kpi_data = [
    (c1, "DATASET", "200", "Pelanggan Mall", "badge-blue", ""),
    (c2, "CLUSTER OPTIMAL", f"{K}", "berdasarkan Silhouette", "badge-blue", ""),
    (c3, "SILHOUETTE SCORE", f"{RES['sil']:.4f}", f"vs baseline {SIL_BASE:.4f}", 
     "badge-green" if imp_sil >= 0 else "badge-orange", 
     f"{'↑' if imp_sil >= 0 else '↓'} {abs(imp_sil):.1f}%"),
    (c4, "WCSS FINAL", f"{RES['wcss_final']:.2f}", f"baseline {WCSS_BASE:.2f}",
     "badge-green" if imp_wcss >= 0 else "badge-orange",
     f"{'↑' if imp_wcss >= 0 else '↓'} {abs(imp_wcss):.1f}%"),
    (c5, "ACCEPTANCE RATE", f"{RES['accept_rate']:.1f}%", 
     f"{RES['iters']} iterasi SA", "badge-blue", ""),
]
for col, label, val, sub, badge_cls, badge_txt in kpi_data:
    with col:
        badge_html = f"<div class='metric-badge {badge_cls}'>{badge_txt}</div>" if badge_txt else ""
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-sub'>{sub}</div>
            {badge_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB NAVIGASI
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📊  Eksplorasi Data  ",
    "  🔬  Analisis PCA  ",
    "  🔥  Simulated Annealing  ",
    "  🎯  Hasil Klasterisasi  ",
    "  ⚖️  Perbandingan Metode  "
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("""<div class='section-header'>
            <div class='section-dot'></div>
            <span class='section-title'>Distribusi Fitur Numerik</span>
            <span class='section-sub'>200 sampel · Mall Customers Dataset</span>
        </div>""", unsafe_allow_html=True)
        
        fig_hist = make_subplots(rows=1, cols=3,
            subplot_titles=['Usia', 'Pendapatan Tahunan (k$)', 'Spending Score'])
        
        for i, col in enumerate(NUM_COLS):
            fig_hist.add_trace(go.Histogram(
                x=df[col], nbinsx=20, name=col.split('(')[0].strip(),
                marker_color=PAL[i], opacity=0.8,
                hovertemplate=f'<b>{col.split("(")[0].strip()}</b><br>Nilai: %{{x}}<br>Frekuensi: %{{y}}<extra></extra>'
            ), row=1, col=i+1)
            fig_hist.add_vline(x=df[col].mean(), line_dash="dash", line_color="white",
                               line_width=1.5, row=1, col=i+1)
            fig_hist.add_vline(x=df[col].median(), line_dash="dot", line_color="#FFA726",
                               line_width=1.5, row=1, col=i+1)
        
        fig_hist.update_layout(**SUBPLOT_THEME,
            height=280, showlegend=False, margin=dict(l=40,r=20,t=40,b=30))
        fig_hist.update_annotations(font_size=11, font_color='#A5B4FC')
        for i in range(1, 4):
            fig_hist.update_xaxes(gridcolor='#1E2235', linecolor='#2A2D3E', row=1, col=i)
            fig_hist.update_yaxes(gridcolor='#1E2235', linecolor='#2A2D3E', row=1, col=i)
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

    with col_r:
        st.markdown("""<div class='section-header'>
            <div class='section-dot'></div>
            <span class='section-title'>Distribusi Gender</span>
        </div>""", unsafe_allow_html=True)
        
        gc = df['Genre'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=gc.index, values=gc.values,
            hole=0.55, marker_colors=['#AB47BC','#42A5F5'],
            textinfo='label+percent', textfont_size=11,
            hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<br>%{percent}<extra></extra>'
        ))
        fig_pie.update_layout(**SUBPLOT_THEME,
            height=240, margin=dict(l=10,r=10,t=20,b=10), showlegend=True,
            legend=dict(orientation='h', y=-0.05, x=0.5, xanchor='center',
                        font=dict(size=11, color='#C5CBE0')))
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""<div class='section-header' style='margin-top:0.5rem;'>
        <div class='section-dot'></div>
        <span class='section-title'>Scatter Plot Hubungan Antar Fitur</span>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    pairs = [
        (col1, 'Age', 'Spending Score (1-100)', PAL[0]),
        (col2, 'Annual Income (k$)', 'Spending Score (1-100)', PAL[1]),
        (col3, 'Age', 'Annual Income (k$)', PAL[2])
    ]
    for container, cx, cy, color in pairs:
        with container:
            fig_sc = go.Figure(go.Scatter(
                x=df[cx], y=df[cy], mode='markers',
                marker=dict(color=color, size=6, opacity=0.6,
                            line=dict(width=0.5, color='rgba(255,255,255,0.2)')),
                text=df['Genre'],
                hovertemplate=f'<b>{cx.split("(")[0].strip()}</b>: %{{x}}<br>'
                              f'<b>{cy.split("(")[0].strip()}</b>: %{{y}}<br>'
                              f'Gender: %{{text}}<extra></extra>'
            ))
            fig_sc.update_layout(**SUBPLOT_THEME,
                height=240, margin=dict(l=45,r=15,t=35,b=35),
                title=dict(text=f'{cx.split("(")[0].strip()} vs {cy.split("(")[0].strip()}',
                           font=dict(size=11, color='#C5CBE0'), x=0.01),
                xaxis=dict(title=cx.split('(')[0].strip(), gridcolor='#1E2235', title_font=dict(size=10)),
                yaxis=dict(title=cy.split('(')[0].strip(), gridcolor='#1E2235', title_font=dict(size=10)))
            st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""<div class='section-header'>
        <div class='section-dot'></div>
        <span class='section-title'>Matriks Korelasi</span>
    </div>""", unsafe_allow_html=True)
    
    corr = df[NUM_COLS].corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=['Usia','Pendapatan','Spending'],
        y=['Usia','Pendapatan','Spending'],
        colorscale=[[0,'#EF5350'],[0.5,'#131825'],[1,'#26A69A']],
        zmin=-1, zmax=1, text=corr.values.round(2),
        texttemplate='%{text}', textfont_size=14,
        hovertemplate='%{y} × %{x}: %{z:.4f}<extra></extra>'
    ))
    fig_corr.update_layout(**SUBPLOT_THEME,
        height=280, margin=dict(l=60,r=60,t=30,b=30), width=400)
    c_corr, c_stat = st.columns([1, 2])
    with c_corr:
        st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
    with c_stat:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.dataframe(
            df[NUM_COLS].describe().round(2).rename(columns={
                'Age':'Usia', 'Annual Income (k$)':'Pendapatan (k$)',
                'Spending Score (1-100)':'Spending Score'
            }),
            use_container_width=True, height=245
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PCA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""<div class='section-header'>
        <div class='section-dot'></div>
        <span class='section-title'>Scree Plot — Explained Variance</span>
        <span class='section-sub'>PC1–PC3 dipilih: {:.2f}% variance tercakup</span>
    </div>""".format(EV_CUM[N_CLU-1]*100), unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_scree = go.Figure()
        colors_bar = [PAL[0] if i < N_CLU else '#2A2D3E' for i in range(len(EV))]
        fig_scree.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(len(EV))],
            y=EV*100, name='Individual', marker_color=colors_bar, opacity=0.9,
            text=[f'{v:.1f}%' for v in EV*100], textposition='outside', textfont=dict(size=9),
            hovertemplate='<b>%{x}</b><br>Variance: %{y:.2f}%<extra></extra>'
        ))
        fig_scree.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(len(EV))], y=EV_CUM*100,
            name='Kumulatif', mode='lines+markers',
            line=dict(color='#FFA726', width=2.5),
            marker=dict(size=7, color='#FFA726'),
            hovertemplate='<b>%{x}</b><br>Kumulatif: %{y:.2f}%<extra></extra>'
        ))
        fig_scree.add_hline(y=80, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                             line_width=1.5, annotation_text="80% threshold",
                             annotation_font_color="#C5CBE0", annotation_font_size=10)
        fig_scree.update_layout(**SUBPLOT_THEME, height=320,
            title='Scree Plot — Explained Variance per Komponen',
            xaxis=dict(gridcolor='#1E2235', linecolor='#2A2D3E'),
            yaxis=dict(title='Variance (%)', gridcolor='#1E2235', linecolor='#2A2D3E'),
            legend=dict(orientation='h', y=1.05, x=0.01, font=dict(size=10)))
        st.plotly_chart(fig_scree, use_container_width=True, config={'displayModeBar': False})

    with col_s2:
        load_df = pd.DataFrame(pca3.components_.T, index=FEATURES_DISPLAY,
                                columns=[f'PC{i+1}' for i in range(N_CLU)])
        fig_heat = go.Figure(go.Heatmap(
            z=pca3.components_,
            x=feat_short, y=[f'PC{i+1}' for i in range(N_CLU)],
            colorscale=[[0,'#EF5350'],[0.5,'#131825'],[1,'#5C6BC0']],
            zmin=-1, zmax=1,
            text=pca3.components_.round(3), texttemplate='%{text}', textfont_size=12,
            hovertemplate='<b>%{y}</b> × <b>%{x}</b>: %{z:.4f}<extra></extra>'
        ))
        fig_heat.update_layout(**SUBPLOT_THEME,
            height=320, title='Heatmap Loading — Kontribusi Fitur per PC',
            xaxis=dict(side='bottom'), margin=dict(l=60,r=60,t=50,b=40))
        st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""<div class='section-header'>
        <div class='section-dot'></div>
        <span class='section-title'>Biplot — Proyeksi Data & Loading Vector</span>
    </div>""", unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig_biplot = go.Figure()
        fig_biplot.add_trace(go.Scatter(
            x=X2[:,0], y=X2[:,1], mode='markers',
            marker=dict(color='#2A3A5C', size=6, opacity=0.5,
                        line=dict(width=0.3, color='rgba(255,255,255,0.1)')),
            name='Data', hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
        ))
        for i, feat in enumerate(feat_short):
            lx = pca2.components_[0,i]*2.8
            ly = pca2.components_[1,i]*2.8
            fig_biplot.add_annotation(
                x=lx, y=ly, ax=0, ay=0,
                arrowhead=3, arrowwidth=2.5, arrowcolor=PAL[i],
                text=f"<b>{feat}</b>", font=dict(color=PAL[i], size=11),
                xanchor='center', showarrow=True
            )
        fig_biplot.update_layout(**SUBPLOT_THEME, height=340,
            title=f'Biplot PC1 ({EV[0]*100:.1f}%) vs PC2 ({EV[1]*100:.1f}%)',
            xaxis=dict(title=f'PC1 ({EV[0]*100:.1f}%)', gridcolor='#1E2235', zeroline=True, zerolinecolor='#2A2D3E'),
            yaxis=dict(title=f'PC2 ({EV[1]*100:.1f}%)', gridcolor='#1E2235', zeroline=True, zerolinecolor='#2A2D3E'),
            showlegend=False)
        st.plotly_chart(fig_biplot, use_container_width=True, config={'displayModeBar': False})

    with col_b2:
        fig_bar_load = go.Figure()
        for i, feat in enumerate(feat_short):
            fig_bar_load.add_trace(go.Bar(
                name=feat, x=[f'PC{j+1}' for j in range(N_CLU)],
                y=pca3.components_[:, i], marker_color=PAL[i], opacity=0.85,
                hovertemplate=f'<b>{feat}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>'
            ))
        fig_bar_load.update_layout(**SUBPLOT_THEME, height=340, barmode='group',
            title='Nilai Loading per Komponen Utama',
            xaxis=dict(gridcolor='#1E2235', linecolor='#2A2D3E'),
            yaxis=dict(title='Loading Value', gridcolor='#1E2235', linecolor='#2A2D3E',
                       zeroline=True, zerolinecolor='#2A2D3E'),
            legend=dict(orientation='h', y=1.05, x=0, font=dict(size=10)))
        st.plotly_chart(fig_bar_load, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""<div class='info-box'>
        <b>Interpretasi PCA:</b> PC1 (%.1f%%) didominasi oleh hubungan terbalik antara Usia dan Spending Score.
        PC2 (%.1f%%) mencerminkan tingkat Annual Income. PC3 (%.1f%%) membedakan berdasarkan Genre.
        Ketiga PC bersama mencakup <b>%.2f%%</b> total variance.
    </div>""" % (EV[0]*100, EV[1]*100, EV[2]*100, EV_CUM[2]*100), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIMULATED ANNEALING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    hist_df = RES['hist']
    iters_x = hist_df['iter'].values

    col_sa1, col_sa2 = st.columns(2)
    
    with col_sa1:
        st.markdown("""<div class='section-header'>
            <div class='section-dot'></div>
            <span class='section-title'>Konvergensi WCSS</span>
        </div>""", unsafe_allow_html=True)
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=iters_x, y=hist_df['wcss_curr'], mode='lines',
            name='WCSS Saat Ini', line=dict(color='#3D4F6B', width=1),
            opacity=0.5, hovertemplate='Iterasi %{x}<br>WCSS: %{y:.4f}<extra></extra>'
        ))
        # Smoothed
        wcss_sm = pd.Series(RES['wcss_log']).rolling(15, min_periods=1).mean()
        fig_conv.add_trace(go.Scatter(
            x=list(range(len(wcss_sm))), y=wcss_sm.values, mode='lines',
            name='Moving Avg (w=15)', line=dict(color=PAL[1], width=2),
            hovertemplate='Iterasi %{x}<br>MA WCSS: %{y:.4f}<extra></extra>'
        ))
        fig_conv.add_trace(go.Scatter(
            x=iters_x, y=hist_df['wcss_best'], mode='lines',
            name='WCSS Terbaik', line=dict(color='#FFA726', width=2.5),
            hovertemplate='Iterasi %{x}<br>Best WCSS: %{y:.4f}<extra></extra>'
        ))
        fig_conv.add_hline(y=WCSS_BASE, line_dash="dash", line_color=PAL[2],
                            line_width=2, annotation_text=f"Baseline K-Means++ ({WCSS_BASE:.2f})",
                            annotation_font_color=PAL[2], annotation_font_size=9)
        fig_conv.add_hline(y=RES['wcss_final'], line_dash="dot", line_color=PAL[1],
                            line_width=1.5, annotation_text=f"SA Final ({RES['wcss_final']:.2f})",
                            annotation_font_color=PAL[1], annotation_font_size=9)
        fig_conv.update_layout(**SUBPLOT_THEME, height=360,
            title=dict(text='Konvergensi WCSS Simulated Annealing', font=dict(size=13), x=0, xanchor='left', y=0.98, yanchor='top'),
            xaxis=dict(title='Iterasi', gridcolor='#1E2235'),
            yaxis=dict(title='WCSS', gridcolor='#1E2235'),
            legend=dict(
                orientation='h', x=0, y=1, xanchor='left', yanchor='bottom',
                font=dict(size=9), bgcolor='rgba(0,0,0,0)', borderwidth=0,
                tracegroupgap=0
            ),
            margin=dict(l=50, r=20, t=80, b=40))
        st.plotly_chart(fig_conv, use_container_width=True, config={'displayModeBar': False})

    with col_sa2:
        st.markdown("""<div class='section-header'>
            <div class='section-dot'></div>
            <span class='section-title'>Cooling Schedule</span>
        </div>""", unsafe_allow_html=True)
        
        fig_cool = go.Figure()
        t_vals = RES['t_log']
        n = len(t_vals)
        fig_cool.add_trace(go.Scatter(
            x=list(range(n)), y=t_vals, mode='lines', fill='tozeroy',
            name='Suhu T', line=dict(color=PAL[2], width=2),
            fillcolor='rgba(239,83,80,0.08)',
            hovertemplate='Iterasi %{x}<br>T: %{y:.6f}<extra></extra>'
        ))
        # Fase annotations
        for start, end, label, color in [(0, int(n*0.2), 'Eksplorasi', PAL[2]),
                                          (int(n*0.2), int(n*0.65), 'Transisi', '#FFA726'),
                                          (int(n*0.65), n, 'Eksploitasi', PAL[1])]:
            fig_cool.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.05, line_width=0)
        fig_cool.update_layout(**SUBPLOT_THEME, height=320,
            title=f'Cooling Schedule (α={alpha})',
            xaxis=dict(title='Iterasi', gridcolor='#1E2235'),
            yaxis=dict(title='Suhu T (log scale)', type='log', gridcolor='#1E2235'),
            showlegend=False)
        st.plotly_chart(fig_cool, use_container_width=True, config={'displayModeBar': False})

    col_sa3, col_sa4 = st.columns(2)
    
    with col_sa3:
        st.markdown("""<div class='section-header'>
            <div class='section-dot'></div>
            <span class='section-title'>Distribusi ΔE</span>
        </div>""", unsafe_allow_html=True)
        
        d_neg = hist_df['delta'][hist_df['delta'] < 0]
        d_pos = hist_df['delta'][hist_df['delta'] > 0]
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Histogram(
            x=d_neg, nbinsx=40, name=f'ΔE<0: Accept ({len(d_neg)})',
            marker_color=PAL[1], opacity=0.85,
            hovertemplate='ΔE: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        fig_delta.add_trace(go.Histogram(
            x=d_pos, nbinsx=40, name=f'ΔE>0: Metropolis ({len(d_pos)})',
            marker_color=PAL[2], opacity=0.85,
            hovertemplate='ΔE: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        fig_delta.add_vline(x=0, line_color="rgba(255,255,255,0.5)", line_width=1.5)
        fig_delta.update_layout(**SUBPLOT_THEME, height=360, barmode='overlay',
            title=dict(text='Distribusi ΔE per Iterasi', font=dict(size=13), x=0, xanchor='left', y=0.98, yanchor='top'),
            xaxis=dict(title='ΔE (Perubahan WCSS)', gridcolor='#1E2235'),
            yaxis=dict(title='Frekuensi', gridcolor='#1E2235'),
            legend=dict(
                orientation='h', x=0, y=1, xanchor='left', yanchor='bottom',
                font=dict(size=9), bgcolor='rgba(0,0,0,0)', borderwidth=0,
                tracegroupgap=0
            ),
            margin=dict(l=50, r=20, t=80, b=40))
        st.plotly_chart(fig_delta, use_container_width=True, config={'displayModeBar': False})

    with col_sa4:
        st.markdown("""<div class='section-header'>
            <div class='section-dot'></div>
            <span class='section-title'>Acceptance Rate per Blok</span>
        </div>""", unsafe_allow_html=True)
        
        block = max(20, len(hist_df)//25)
        acc_blk, centers_blk = [], []
        for i in range(0, len(hist_df), block):
            chunk = hist_df.iloc[i:i+block]
            n_acc = chunk['status'].str.startswith('Accept').sum()
            acc_blk.append(n_acc / len(chunk) * 100)
            centers_blk.append(i + block/2)
        
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=centers_blk, y=acc_blk, name='Acceptance Rate',
            marker_color=PAL[0], opacity=0.8,
            hovertemplate='Iterasi %{x:.0f}<br>Rate: %{y:.1f}%<extra></extra>'
        ))
        fig_acc.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
        fig_acc.update_layout(**SUBPLOT_THEME, height=360,
            title=dict(text='Acceptance Rate per Blok Iterasi', font=dict(size=13), x=0, xanchor='left'),
            xaxis=dict(title='Iterasi', gridcolor='#1E2235'),
            yaxis=dict(title='Rate (%)', range=[0,105], gridcolor='#1E2235'),
            margin=dict(l=50, r=20, t=40, b=40))
        st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})

    # Ringkasan SA
    st.markdown("""<div class='section-header'>
        <div class='section-dot'></div>
        <span class='section-title'>Ringkasan Simulated Annealing</span>
    </div>""", unsafe_allow_html=True)
    
    total_moves = RES['n_better'] + RES['n_metro'] + RES['n_reject']
    sa_metrics = {
        'Total Iterasi': RES['iters'],
        'Waktu (detik)': f"{RES['time']:.3f}",
        'Accept ΔE<0': f"{RES['n_better']} ({RES['n_better']/total_moves*100:.1f}%)",
        'Accept Metropolis': f"{RES['n_metro']} ({RES['n_metro']/total_moves*100:.1f}%)",
        'Reject': f"{RES['n_reject']} ({RES['n_reject']/total_moves*100:.1f}%)",
        'Escape Local Min': RES['n_escape'],
        'WCSS Inisialisasi': f"{RES['wcss_init']:.4f}",
        'WCSS Terbaik SA': f"{RES['wcss_best_sa']:.4f}",
        'WCSS Final': f"{RES['wcss_final']:.4f}",
    }
    cols_sa = st.columns(3)
    items = list(sa_metrics.items())
    for idx, (k, v) in enumerate(items):
        with cols_sa[idx % 3]:
            st.markdown(f"""
            <div style='background:#131825; border:1px solid #1E2235; border-radius:8px;
                        padding:0.7rem 1rem; margin-bottom:0.5rem;'>
                <div style='font-size:0.72rem; color:#5C6A8A; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;'>{k}</div>
                <div style='font-size:1.05rem; font-weight:600; color:#C5CBE0; margin-top:0.2rem;'>{v}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HASIL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    # K optimal
    st.markdown("""<div class='section-header'>
        <div class='section-dot'></div>
        <span class='section-title'>Pemilihan K Optimal</span>
    </div>""", unsafe_allow_html=True)
    
    kl = list(df_k.index)
    fig_k = make_subplots(rows=1, cols=4, subplot_titles=['WCSS (↓)','Silhouette (↑)','Davies-Bouldin (↓)','Calinski-H (↑)'])
    metrics_k = [('WCSS', PAL[0], True), ('Silhouette', PAL[1], False),
                 ('Davies-Bouldin', PAL[2], True), ('Calinski-H', PAL[3], False)]
    for i, (m, color, low) in enumerate(metrics_k):
        vals = df_k[m].tolist()
        opt_i = np.argmin(vals) if low else np.argmax(vals)
        fig_k.add_trace(go.Scatter(
            x=kl, y=vals, mode='lines+markers', name=m,
            line=dict(color=color, width=2.2),
            marker=dict(size=[10 if j == opt_i else 5 for j in range(len(kl))],
                        color=[color]*len(kl),
                        line=dict(width=[2 if j == opt_i else 0 for j in range(len(kl))],
                                  color='white')),
            hovertemplate=f'K=%{{x}}<br>{m}: %{{y:.4f}}<extra></extra>'
        ), row=1, col=i+1)
        fig_k.add_vline(x=kl[opt_i], line_dash="dot", line_color='rgba(255,255,255,0.25)',
                         line_width=1.2, row=1, col=i+1)
    fig_k.update_layout(**SUBPLOT_THEME,
        height=280, showlegend=False, margin=dict(l=40,r=20,t=40,b=30))
    fig_k.update_annotations(font_size=11, font_color='#A5B4FC')
    for i in range(1, 5):
        fig_k.update_xaxes(gridcolor='#1E2235', linecolor='#2A2D3E', row=1, col=i)
        fig_k.update_yaxes(gridcolor='#1E2235', linecolor='#2A2D3E', row=1, col=i)
    st.plotly_chart(fig_k, use_container_width=True, config={'displayModeBar': False})

    # Scatter cluster 2D & 3D
    cl2, cl3 = st.columns(2)
    cents_2d = pca2.transform(pca3.inverse_transform(RES['centers']))
    
    with cl2:
        st.markdown("""<div class='section-header'>
            <div class='section-dot'></div>
            <span class='section-title'>Proyeksi Cluster 2D (PC1 vs PC2)</span>
        </div>""", unsafe_allow_html=True)
        fig_2d = go.Figure()
        for c in cluster_ids:
            m = RES['labels'] == c
            fig_2d.add_trace(go.Scatter(
                x=X2[m,0], y=X2[m,1], mode='markers',
                name=f'Cluster {c} (n={m.sum()})',
                marker=dict(color=PAL[c%len(PAL)], size=7, opacity=0.8,
                            line=dict(width=0.5, color='rgba(255,255,255,0.2)')),
                hovertemplate=f'<b>Cluster {c}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>'
            ))
        fig_2d.add_trace(go.Scatter(
            x=cents_2d[:,0], y=cents_2d[:,1], mode='markers',
            name='Centroid', marker=dict(symbol='star', size=16, color='white',
                                          line=dict(width=1.5, color='black')),
            hovertemplate='<b>Centroid</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
        ))
        fig_2d.update_layout(**SUBPLOT_THEME, height=350,
            xaxis=dict(title=f'PC1 ({EV[0]*100:.1f}%)', gridcolor='#1E2235'),
            yaxis=dict(title=f'PC2 ({EV[1]*100:.1f}%)', gridcolor='#1E2235'),
            legend=dict(
                font=dict(size=9, color='#C5CBE0'),
                bgcolor='rgba(13,15,26,0.85)',
                bordercolor='#2A2D3E',
                borderwidth=1,
                orientation='v',
                x=1.01, y=1,
                xanchor='left', yanchor='top',
                itemsizing='constant',
                tracegroupgap=2,
            ),
            margin=dict(l=50, r=130, t=30, b=50))
        st.plotly_chart(fig_2d, use_container_width=True, config={'displayModeBar': False})

    with cl3:
        st.markdown("""<div class='section-header'>
            <div class='section-dot'></div>
            <span class='section-title'>Proyeksi Cluster 3D (PC1-PC2-PC3)</span>
        </div>""", unsafe_allow_html=True)
        fig_3d = go.Figure()
        for c in cluster_ids:
            m = RES['labels'] == c
            fig_3d.add_trace(go.Scatter3d(
                x=X3[m,0], y=X3[m,1], z=X3[m,2],
                mode='markers', name=f'Cluster {c}',
                marker=dict(color=PAL[c%len(PAL)], size=4, opacity=0.75,
                            line=dict(width=0.3, color='rgba(255,255,255,0.1)')),
                hovertemplate=f'<b>Cluster {c}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>PC3: %{{z:.3f}}<extra></extra>'
            ))
        fig_3d.add_trace(go.Scatter3d(
            x=RES['centers'][:,0], y=RES['centers'][:,1], z=RES['centers'][:,2],
            mode='markers', name='Centroid',
            marker=dict(symbol='diamond', size=8, color='white', opacity=1)
        ))
        fig_3d.update_layout(
            paper_bgcolor='#0A0C14', font=dict(color='#C5CBE0', family='Inter'),
            height=350, margin=dict(l=0,r=0,t=20,b=0),
            scene=dict(
                bgcolor='#131825',
                xaxis=dict(title='PC1', gridcolor='#2A2D3E', backgroundcolor='#131825', color='#7986CB'),
                yaxis=dict(title='PC2', gridcolor='#2A2D3E', backgroundcolor='#131825', color='#7986CB'),
                zaxis=dict(title='PC3', gridcolor='#2A2D3E', backgroundcolor='#131825', color='#7986CB')
            ),
            legend=dict(font=dict(size=10), bgcolor='rgba(19,24,37,0.8)')
        )
        st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': False})

    # Profil Cluster
    st.markdown("""<div class='section-header' style='margin-top:0.5rem;'>
        <div class='section-dot'></div>
        <span class='section-title'>Profil & Interpretasi Cluster</span>
    </div>""", unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns([1.5, 1.5, 1])
    
    with pc1:
        fig_bar_prof = go.Figure()
        for i, col in enumerate(NUM_COLS):
            means = [profile.loc[c, col] for c in cluster_ids]
            fig_bar_prof.add_trace(go.Bar(
                name=col.replace(' (k$)','').replace(' (1-100)',''),
                x=[f'Cluster {c}' for c in cluster_ids], y=means,
                marker_color=PAL[i], opacity=0.85,
                hovertemplate=f'<b>%{{x}}</b><br>{col}: %{{y:.2f}}<extra></extra>'
            ))
        fig_bar_prof.update_layout(**SUBPLOT_THEME, height=360, barmode='group',
            title=dict(text='Rata-rata Fitur per Cluster', font=dict(size=13), x=0, xanchor='left', y=0.98, yanchor='top'),
            xaxis=dict(gridcolor='#1E2235'), yaxis=dict(title='Nilai Rata-rata', gridcolor='#1E2235'),
            legend=dict(
                orientation='h', x=0, y=1, xanchor='left', yanchor='bottom',
                font=dict(size=10), bgcolor='rgba(0,0,0,0)', borderwidth=0,
                tracegroupgap=0
            ),
            margin=dict(l=50, r=20, t=80, b=40))
        st.plotly_chart(fig_bar_prof, use_container_width=True, config={'displayModeBar': False})
    
    with pc2:
        fig_viol = make_subplots(rows=1, cols=3,
            subplot_titles=['Usia','Pendapatan','Spending'])
        for i, col in enumerate(NUM_COLS):
            for c in cluster_ids:
                data_c = df[df['Cluster']==c][col].values
                fig_viol.add_trace(go.Violin(
                    y=data_c, x=[f'C{c}']*len(data_c),
                    name=f'C{c}', fillcolor=PAL[c%len(PAL)],
                    opacity=0.75, line_color='rgba(255,255,255,0.2)',
                    meanline_visible=True, showlegend=(i==0),
                    hovertemplate=f'Cluster {c}<br>{col}: %{{y:.1f}}<extra></extra>'
                ), row=1, col=i+1)
        fig_viol.update_layout(**SUBPLOT_THEME,
            height=280, showlegend=False, margin=dict(l=40,r=20,t=40,b=30))
        fig_viol.update_annotations(font_size=10, font_color='#A5B4FC')
        for i in range(1, 4):
            fig_viol.update_xaxes(gridcolor='#1E2235', linecolor='#2A2D3E', row=1, col=i)
            fig_viol.update_yaxes(gridcolor='#1E2235', linecolor='#2A2D3E', row=1, col=i)
        st.plotly_chart(fig_viol, use_container_width=True, config={'displayModeBar': False})

    with pc3:
        # Radar chart interaktif
        cats = ['Usia','Pendapatan','Spending']
        rd = profile[NUM_COLS].copy()
        for col in rd.columns:
            rd[col] = (rd[col]-rd[col].min())/(rd[col].max()-rd[col].min()+1e-9)
        
        fig_radar = go.Figure()
        for c in cluster_ids:
            vals_r = rd.loc[c].tolist()
            vals_r += vals_r[:1]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_r, theta=cats+[cats[0]], fill='toself',
                name=f'Cluster {c}', line=dict(color=PAL[c%len(PAL)], width=2),
                fillcolor=f'rgba({",".join(str(int(PAL[c%len(PAL)].lstrip("#")[i:i+2], 16)) for i in (0,2,4))},0.15)',
                hovertemplate=f'<b>Cluster {c}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>'
            ))
        fig_radar.update_layout(
            paper_bgcolor='#0A0C14', plot_bgcolor='#131825',
            font=dict(color='#C5CBE0', family='Inter'),
            height=280, margin=dict(l=20,r=20,t=40,b=20),
            polar=dict(
                bgcolor='#131825',
                radialaxis=dict(visible=True, range=[0,1], gridcolor='#2A2D3E', color='#5C6A8A'),
                angularaxis=dict(gridcolor='#2A2D3E', color='#A5B4FC')
            ),
            legend=dict(font=dict(size=10), bgcolor='rgba(19,24,37,0.8)'),
            title=dict(text='Radar Chart Cluster', font=dict(size=12, color='#C5CBE0'), x=0.01)
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    # Tabel profil
    profile_display = profile.copy()
    profile_display.columns = ['Usia Rata-rata','Pendapatan (k$)','Spending Score','Jumlah','Proporsi (%)']
    profile_display.index = [f'Cluster {i}' for i in profile_display.index]
    
    # Interpretasi
    st.markdown("**Interpretasi Segmen**")
    seg_cols = st.columns(K)
    for idx, c in enumerate(cluster_ids):
        r = profile.loc[c]
        inc = 'Tinggi' if r['Annual Income (k$)'] > 65 else ('Sedang' if r['Annual Income (k$)'] > 42 else 'Rendah')
        sco = 'Tinggi' if r['Spending Score (1-100)'] > 60 else ('Sedang' if r['Spending Score (1-100)'] > 40 else 'Rendah')
        age_lbl = 'Tua' if r['Age'] > 50 else ('Muda' if r['Age'] < 30 else 'Dewasa')
        color_hex = PAL[c%len(PAL)]
        with seg_cols[idx]:
            st.markdown(f"""
            <div style='background:#131825; border:1px solid #1E2235; border-top:2px solid {color_hex};
                        border-radius:10px; padding:1rem; text-align:center;'>
                <div style='font-size:1.2rem; font-weight:700; color:{color_hex};'>Cluster {c}</div>
                <div style='font-size:1.4rem; font-weight:700; color:#E8EAF6; margin:0.3rem 0;'>
                    {int(r["N"])} <span style='font-size:0.85rem; color:#5C6A8A; font-weight:400;'>pelanggan</span>
                </div>
                <div style='font-size:0.78rem; color:#7986CB; margin-bottom:0.5rem;'>{r["%"]:.1f}% dari total</div>
                <div style='background:#0A0C14; border-radius:6px; padding:0.6rem; text-align:left; font-size:0.8rem;'>
                    <div style='color:#C5CBE0; margin-bottom:0.3rem;'>
                        📅 Usia rata-rata: <b style='color:#E8EAF6;'>{r["Age"]:.1f} th</b> ({age_lbl})
                    </div>
                    <div style='color:#C5CBE0; margin-bottom:0.3rem;'>
                        💰 Pendapatan: <b style='color:#E8EAF6;'>{r["Annual Income (k$)"]:.1f}k$</b> ({inc})
                    </div>
                    <div style='color:#C5CBE0;'>
                        🛒 Spending: <b style='color:#E8EAF6;'>{r["Spending Score (1-100)"]:.1f}/100</b> ({sco})
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PERBANDINGAN METODE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    km_rand = KMeans(n_clusters=K, init='random', n_init=30, random_state=42).fit(X3)
    km_kpp  = KMeans(n_clusters=K, init='k-means++', n_init=30, random_state=42).fit(X3)
    
    km_r_wcss = km_rand.inertia_
    km_r_sil  = silhouette_score(X3, km_rand.labels_)
    km_r_db   = davies_bouldin_score(X3, km_rand.labels_)
    km_r_ch   = calinski_harabasz_score(X3, km_rand.labels_)
    
    methods_data = {
        'Metode': ['K-Means Random', 'K-Means++', 'SA + K-Means (Hybrid)'],
        'WCSS': [km_r_wcss, WCSS_BASE, RES['wcss_final']],
        'Silhouette': [km_r_sil, SIL_BASE, RES['sil']],
        'Davies-Bouldin': [km_r_db, DB_BASE, RES['db']],
        'Calinski-H': [km_r_ch, silhouette_score(X3, km_kpp.labels_)*100, RES['ch']]
    }
    
    st.markdown("""<div class='section-header'>
        <div class='section-dot'></div>
        <span class='section-title'>Perbandingan Kinerja Metode</span>
    </div>""", unsafe_allow_html=True)
    
    m_colors = [PAL[3], PAL[0], PAL[1]]
    comp1, comp2, comp3 = st.columns(3)
    
    for container, metric, lower_better in [(comp1, 'WCSS', True),
                                             (comp2, 'Silhouette', False),
                                             (comp3, 'Davies-Bouldin', True)]:
        with container:
            vals_m = [km_r_wcss if metric=='WCSS' else (km_r_sil if metric=='Silhouette' else km_r_db),
                      WCSS_BASE if metric=='WCSS' else (SIL_BASE if metric=='Silhouette' else DB_BASE),
                      RES['wcss_final'] if metric=='WCSS' else (RES['sil'] if metric=='Silhouette' else RES['db'])]
            best_i = np.argmin(vals_m) if lower_better else np.argmax(vals_m)
            border_widths = [2.5 if i == best_i else 0.5 for i in range(3)]
            
            fig_comp = go.Figure(go.Bar(
                x=['K-Means\nRandom','K-Means\n++','SA+K-Means'],
                y=vals_m, marker_color=m_colors, opacity=0.85,
                text=[f'{v:.4f}' for v in vals_m], textposition='outside', textfont=dict(size=10),
                marker_line_width=border_widths, marker_line_color='white',
                hovertemplate='<b>%{x}</b><br>' + metric + ': %{y:.4f}<extra></extra>'
            ))
            arrow = '↓' if lower_better else '↑'
            fig_comp.update_layout(**SUBPLOT_THEME, height=280,
                title=f'{metric} ({arrow} lebih baik)',
                xaxis=dict(gridcolor='#1E2235'),
                yaxis=dict(title=metric, gridcolor='#1E2235'),
                showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})

    # Summary table
    st.markdown("""<div class='section-header'>
        <div class='section-dot'></div>
        <span class='section-title'>Tabel Perbandingan Lengkap</span>
    </div>""", unsafe_allow_html=True)
    
    imp_wcss_v = (WCSS_BASE - RES['wcss_final']) / WCSS_BASE * 100
    imp_sil_v  = (RES['sil'] - SIL_BASE) / SIL_BASE * 100
    imp_db_v   = (DB_BASE - RES['db']) / DB_BASE * 100
    
    summary_df = pd.DataFrame({
        'Metode': ['K-Means Random Init', 'K-Means++ Init', 'SA + K-Means (Hybrid)'],
        'WCSS': [f'{km_r_wcss:.4f}', f'{WCSS_BASE:.4f}', f'{RES["wcss_final"]:.4f}'],
        'Silhouette ↑': [f'{km_r_sil:.4f}', f'{SIL_BASE:.4f}', f'{RES["sil"]:.4f}'],
        'Davies-Bouldin ↓': [f'{km_r_db:.4f}', f'{DB_BASE:.4f}', f'{RES["db"]:.4f}'],
        'Calinski-H ↑': [f'{km_r_ch:.2f}', f'{calinski_harabasz_score(X3,km_kpp.labels_):.2f}', f'{RES["ch"]:.2f}'],
        'Ket.': ['—', '← Baseline', '✓ SA Hybrid']
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True, height=130)
    
    # Delta perbaikan SA
    col_imp1, col_imp2, col_imp3 = st.columns(3)
    for container, label, val, better in [
        (col_imp1, 'Perbaikan WCSS vs K-Means++', imp_wcss_v, imp_wcss_v > 0),
        (col_imp2, 'Perbaikan Silhouette vs K-Means++', imp_sil_v, imp_sil_v > 0),
        (col_imp3, 'Perbaikan Davies-Bouldin vs K-Means++', imp_db_v, imp_db_v > 0),
    ]:
        with container:
            color = '#4ade80' if better else '#f87171'
            icon = '↑ Membaik' if better else '↓ Memburuk'
            st.markdown(f"""
            <div style='background:#131825; border:1px solid #1E2235; border-radius:10px;
                        padding:1rem; text-align:center;'>
                <div style='font-size:0.72rem; color:#5C6A8A; text-transform:uppercase;
                            letter-spacing:0.06em; margin-bottom:0.4rem;'>{label}</div>
                <div style='font-size:1.6rem; font-weight:700; color:{color};'>
                    {'+' if val > 0 else ''}{val:.2f}%
                </div>
                <div style='font-size:0.78rem; color:{color}; opacity:0.7; margin-top:0.2rem;'>{icon}</div>
            </div>
            """, unsafe_allow_html=True)

    # Sensitivitas
    st.markdown("""<div class='section-header' style='margin-top:1.5rem;'>
        <div class='section-dot'></div>
        <span class='section-title'>Analisis Sensitivitas Parameter SA (α × σ)</span>
        <span class='section-sub'>9 kombinasi · Fixed T₀=100, MaxIter=600</span>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Menghitung sensitivitas..."):
        ALPHAS = [0.85, 0.92, 0.97]
        SIGMAS = [0.3, 0.5, 0.8]
        
        @st.cache_data
        def compute_sensitivity(K_val, _X3=None):
            SENS = []
            for a, s in itertools.product(ALPHAS, SIGMAS):
                r = run_sa_cached(X3, K_val, T0=100.0, alpha=a, T_min=1e-3,
                                   max_iter=600, sigma=s, seed=42)
                r.update({'alpha': a, 'sigma': s})
                SENS.append(r)
            return SENS
        
        SENS = compute_sensitivity(K)

    heat_wcss = np.array([[r['wcss_final'] for r in SENS if r['alpha'] == a] for a in ALPHAS])
    heat_sil  = np.array([[r['sil']        for r in SENS if r['alpha'] == a] for a in ALPHAS])
    
    sh1, sh2 = st.columns(2)
    with sh1:
        fig_hw = go.Figure(go.Heatmap(
            z=heat_wcss, x=[f'σ={s}' for s in SIGMAS], y=[f'α={a}' for a in ALPHAS],
            colorscale=[[0,'#26A69A'],[0.5,'#131825'],[1,'#EF5350']],
            text=[[f'{v:.3f}\n{"✓" if v < WCSS_BASE else "✗"}' for v in row] for row in heat_wcss],
            texttemplate='%{text}', textfont_size=12,
            hovertemplate='<b>%{y}, %{x}</b><br>WCSS: %{z:.4f}<extra></extra>'
        ))
        fig_hw.update_layout(**SUBPLOT_THEME,
            height=260, title=f'Heatmap WCSS Final: α × σ (baseline={WCSS_BASE:.3f})',
            margin=dict(l=60,r=60,t=50,b=40),
            xaxis=dict(side='bottom'), yaxis=dict(side='left'))
        st.plotly_chart(fig_hw, use_container_width=True, config={'displayModeBar': False})

    with sh2:
        fig_hs = go.Figure(go.Heatmap(
            z=heat_sil, x=[f'σ={s}' for s in SIGMAS], y=[f'α={a}' for a in ALPHAS],
            colorscale=[[0,'#131825'],[0.5,'#5C6BC0'],[1,'#26A69A']],
            text=[[f'{v:.4f}' for v in row] for row in heat_sil],
            texttemplate='%{text}', textfont_size=12,
            hovertemplate='<b>%{y}, %{x}</b><br>Silhouette: %{z:.4f}<extra></extra>'
        ))
        fig_hs.update_layout(**SUBPLOT_THEME,
            height=260, title='Heatmap Silhouette: α × σ (↑ terbaik)',
            margin=dict(l=60,r=60,t=50,b=40),
            xaxis=dict(side='bottom'))
        st.plotly_chart(fig_hs, use_container_width=True, config={'displayModeBar': False})

    best_s = min(SENS, key=lambda x: x['wcss_final'])
    st.markdown(f"""
    <div class='success-box'>
        <b>Kombinasi Parameter Terbaik:</b> α = {best_s['alpha']}, σ = {best_s['sigma']} — 
        WCSS Final: <b>{best_s['wcss_final']:.4f}</b> · 
        Silhouette: <b>{best_s['sil']:.4f}</b> · 
        Escape local minimum: <b>{best_s['n_escape']} kali</b>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#3A4260; font-size:0.78rem; padding:0.5rem 0 1rem;'>
    Mall Customer Segmentation Dashboard &nbsp;·&nbsp; PCA + Simulated Annealing + K-Means
    &nbsp;·&nbsp; 200 samples · 4 features
</div>
""", unsafe_allow_html=True)
