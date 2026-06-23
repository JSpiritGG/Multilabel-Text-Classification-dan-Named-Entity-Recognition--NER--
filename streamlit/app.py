"""
==========================================================================
  ABSA dari Review Google Places:
  Multilabel Text Classification & Named Entity Recognition (NER)
==========================================================================
  Streamlit Application
  Mata Kuliah: Pemrosesan Bahasa Alami - UAS
==========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# ─────────────────── Page Config ───────────────────
st.set_page_config(
    page_title="ABSA - Multilabel Classification & NER",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── Custom CSS ───────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ─── Global ─── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Sidebar styling ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #e0e0ff !important;
    }

    /* ─── Hero header ─── */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    .hero-container h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        position: relative;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .hero-container p {
        font-size: 1.1rem;
        opacity: 0.9;
        position: relative;
        font-weight: 300;
    }

    /* ─── Metric cards ─── */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.8);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #666;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* ─── Aspect badge ─── */
    .aspect-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        color: white;
        letter-spacing: 0.5px;
    }
    .badge-fuel { background: linear-gradient(135deg, #f093fb, #f5576c); }
    .badge-machine { background: linear-gradient(135deg, #4facfe, #00f2fe); }
    .badge-others { background: linear-gradient(135deg, #43e97b, #38f9d7); }
    .badge-part { background: linear-gradient(135deg, #fa709a, #fee140); }
    .badge-price { background: linear-gradient(135deg, #a18cd1, #fbc2eb); }
    .badge-service { background: linear-gradient(135deg, #ffecd2, #fcb69f); color: #333; }

    /* ─── Sentiment chips ─── */
    .sentiment-positive {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 0.25rem 0.7rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
        padding: 0.25rem 0.7rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #bdc3c7, #95a5a6);
        color: white;
        padding: 0.25rem 0.7rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* ─── NER entity highlights ─── */
    .ner-entity {
        display: inline;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
        margin: 0 2px;
    }
    .ner-brand { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
    .ner-product { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .ner-location { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .ner-aspect { background: #f3e8ff; color: #6b21a8; border: 1px solid #c084fc; }

    /* ─── Section headers ─── */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.5rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea 0%, #764ba2 100%) 1;
    }

    /* ─── Info box ─── */
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #312e81;
    }

    /* ─── Prediction result card ─── */
    .prediction-card {
        background: white;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .prediction-card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    /* ─── Table styling ─── */
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #9ca3af;
        font-size: 0.8rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────── Data Loading ───────────────────
@st.cache_data
def load_data():
    """Load preprocessed datasets."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_df = pd.read_csv(os.path.join(base_path, 'train_preprocess.csv'))
    valid_df = pd.read_csv(os.path.join(base_path, 'valid_preprocess.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'test_preprocess.csv'))
    return train_df, valid_df, test_df


ASPECT_COLUMNS = ['fuel', 'machine', 'others', 'part', 'price', 'service']
SENTIMENT_LABELS = ['positive', 'negative', 'neutral']

ASPECT_DESCRIPTIONS = {
    'fuel': '⛽ Bahan Bakar (Fuel)',
    'machine': '⚙️ Mesin (Machine)',
    'others': '🚗 Keseluruhan (Others)',
    'part': '🔧 Bagian/Fitur (Part)',
    'price': '💰 Harga (Price)',
    'service': '🛠️ Layanan (Service)',
}

ASPECT_COLORS = {
    'fuel': '#f5576c',
    'machine': '#4facfe',
    'others': '#43e97b',
    'part': '#fa709a',
    'price': '#a18cd1',
    'service': '#fcb69f',
}


# ─────────────────── Sidebar ───────────────────
with st.sidebar:
    st.markdown("## 🔍 ABSA Navigator")
    st.markdown("---")

    page = st.radio(
        "📑 Pilih Halaman",
        [
            "🏠 Beranda",
            "📊 Eksplorasi Data",
            "🏷️ Multilabel Classification",
            "📛 Named Entity Recognition",
            "🚀 Demo Prediksi",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; opacity: 0.7; font-size: 0.75rem;'>
        <p>📚 Pemrosesan Bahasa Alami</p>
        <p>Universitas Atma Jaya</p>
        <p>Semester 6 • 2026</p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE: BERANDA
# ════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.markdown("""
    <div class="hero-container">
        <h1>🔍 Aspect-Based Sentiment Analysis</h1>
        <p>Pemodelan ABSA dari Review Google Places:<br>
        <strong>Multilabel Text Classification</strong> & <strong>Named Entity Recognition (NER)</strong></p>
    </div>
    """, unsafe_allow_html=True)

    try:
        train_df, valid_df, test_df = load_data()
        total_data = len(train_df) + len(valid_df) + len(test_df)
    except Exception:
        total_data = "N/A"
        train_df = None

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_data}</div>
            <div class="metric-label">Total Review</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">Aspek Kategori</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">Label Sentimen</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">2</div>
            <div class="metric-label">Model NLP</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # About the project
    st.markdown('<div class="section-header">📋 Tentang Proyek</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Aspect-Based Sentiment Analysis (ABSA)</strong> adalah teknik analisis sentimen yang lebih granular,
        yang tidak hanya menentukan sentimen keseluruhan dari sebuah teks, tetapi juga mengidentifikasi
        aspek-aspek spesifik yang dibicarakan dan sentimen yang terkait dengan setiap aspek tersebut.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🏷️ Multilabel Text Classification")
        st.markdown("""
        Setiap review diklasifikasikan ke dalam **6 aspek** dengan **3 sentimen**:

        - ⛽ **Fuel** — Konsumsi bahan bakar
        - ⚙️ **Machine** — Performa mesin
        - 🚗 **Others** — Keseluruhan kendaraan
        - 🔧 **Part** — Bagian/fitur kendaraan
        - 💰 **Price** — Harga & nilai ekonomis
        - 🛠️ **Service** — Layanan bengkel/dealer
        """)

    with col_b:
        st.markdown("#### 📛 Named Entity Recognition (NER)")
        st.markdown("""
        Identifikasi entitas bernama dalam review:

        - 🏭 **Brand** — Merek kendaraan (Toyota, Honda, dll.)
        - 🚙 **Product** — Nama produk (Avanza, Brio, dll.)
        - 📍 **Location** — Lokasi yang disebutkan
        - 🏷️ **Aspect Term** — Istilah aspek spesifik

        Menggunakan pendekatan *rule-based* dan *pattern matching*.
        """)

    st.markdown("---")

    # Pipeline overview
    st.markdown('<div class="section-header">🔄 Pipeline Pemrosesan</div>', unsafe_allow_html=True)

    pipeline_cols = st.columns(5)
    steps = [
        ("📥", "Data\nCollection", "Scraping review Google Places"),
        ("🧹", "Pre-\nprocessing", "Cleaning, tokenisasi, stemming"),
        ("📊", "Feature\nExtraction", "TF-IDF, Word2Vec embeddings"),
        ("🤖", "Model\nTraining", "SVM, Random Forest, dll."),
        ("📈", "Evaluation\n& Demo", "Metrics, prediksi interaktif"),
    ]
    for col, (icon, title, desc) in zip(pipeline_cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:1rem; background:white; border-radius:12px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.06); height:180px; display:flex;
                        flex-direction:column; justify-content:center; align-items:center;
                        border: 1px solid #e5e7eb;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:700; font-size:0.9rem; color:#312e81; white-space:pre-line;">{title}</div>
                <div style="font-size:0.7rem; color:#6b7280; margin-top:0.4rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Aspect badges
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏷️ Aspek Kategori</div>', unsafe_allow_html=True)

    badges_html = ""
    for aspect, desc in ASPECT_DESCRIPTIONS.items():
        badges_html += f'<span class="aspect-badge badge-{aspect}">{desc}</span> '
    st.markdown(badges_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE: EKSPLORASI DATA
# ════════════════════════════════════════════════════════
elif page == "📊 Eksplorasi Data":
    st.markdown("""
    <div class="hero-container" style="padding: 1.5rem;">
        <h1 style="font-size:1.8rem;">📊 Eksplorasi Data</h1>
        <p>Analisis dan visualisasi dataset ABSA</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        train_df, valid_df, test_df = load_data()
    except FileNotFoundError:
        st.error("⚠️ Dataset tidak ditemukan. Pastikan file CSV berada di folder `data/`.")
        st.stop()

    # Dataset overview
    tab1, tab2, tab3 = st.tabs(["📁 Dataset Overview", "📊 Distribusi Sentimen", "☁️ Word Analysis"])

    with tab1:
        st.markdown('<div class="section-header">📁 Ringkasan Dataset</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(train_df)}</div>
                <div class="metric-label">Training Data</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(valid_df)}</div>
                <div class="metric-label">Validation Data</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(test_df)}</div>
                <div class="metric-label">Test Data</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Show sample data
        dataset_choice = st.selectbox("Pilih dataset:", ["Training", "Validation", "Test"])
        if dataset_choice == "Training":
            display_df = train_df
        elif dataset_choice == "Validation":
            display_df = valid_df
        else:
            display_df = test_df

        st.dataframe(display_df.head(20), use_container_width=True, height=400)

        # Sentence length distribution
        st.markdown('<div class="section-header">📏 Distribusi Panjang Kalimat</div>', unsafe_allow_html=True)
        import plotly.express as px
        import plotly.graph_objects as go

        train_df['word_count'] = train_df['sentence'].apply(lambda x: len(str(x).split()))

        fig = px.histogram(
            train_df, x='word_count', nbins=30,
            color_discrete_sequence=['#667eea'],
            labels={'word_count': 'Jumlah Kata', 'count': 'Frekuensi'},
            title='Distribusi Jumlah Kata per Review (Training Set)'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            title_font=dict(size=16, color='#312e81'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">📊 Distribusi Sentimen per Aspek</div>', unsafe_allow_html=True)

        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Count sentiment distribution for each aspect
        sentiment_data = []
        for aspect in ASPECT_COLUMNS:
            counts = train_df[aspect].value_counts()
            for sentiment in SENTIMENT_LABELS:
                sentiment_data.append({
                    'Aspek': ASPECT_DESCRIPTIONS.get(aspect, aspect),
                    'Sentimen': sentiment.capitalize(),
                    'Jumlah': counts.get(sentiment, 0)
                })

        sentiment_df = pd.DataFrame(sentiment_data)

        color_map = {
            'Positive': '#00b09b',
            'Negative': '#eb3349',
            'Neutral': '#95a5a6'
        }

        fig = px.bar(
            sentiment_df, x='Aspek', y='Jumlah', color='Sentimen',
            barmode='group',
            color_discrete_map=color_map,
            title='Distribusi Sentimen per Aspek (Training Set)'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            title_font=dict(size=16, color='#312e81'),
            xaxis_tickangle=-25,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap of non-neutral sentiments
        st.markdown('<div class="section-header">🔥 Heatmap Sentimen Non-Neutral</div>', unsafe_allow_html=True)

        heatmap_data = []
        for aspect in ASPECT_COLUMNS:
            pos_count = (train_df[aspect] == 'positive').sum()
            neg_count = (train_df[aspect] == 'negative').sum()
            neu_count = (train_df[aspect] == 'neutral').sum()
            heatmap_data.append([pos_count, neg_count, neu_count])

        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[ASPECT_DESCRIPTIONS[a] for a in ASPECT_COLUMNS],
            columns=['Positive', 'Negative', 'Neutral']
        )

        fig = px.imshow(
            heatmap_df,
            color_continuous_scale='RdYlGn',
            text_auto=True,
            aspect='auto',
            title='Heatmap Distribusi Sentimen'
        )
        fig.update_layout(
            font=dict(family='Inter'),
            title_font=dict(size=16, color='#312e81'),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Multilabel co-occurrence
        st.markdown('<div class="section-header">🔗 Ko-okurensi Aspek (Non-Neutral)</div>', unsafe_allow_html=True)

        binary_df = pd.DataFrame()
        for aspect in ASPECT_COLUMNS:
            binary_df[aspect] = (train_df[aspect] != 'neutral').astype(int)

        co_occurrence = binary_df.T.dot(binary_df)

        fig = px.imshow(
            co_occurrence,
            color_continuous_scale='Purples',
            text_auto=True,
            x=[ASPECT_DESCRIPTIONS[a] for a in ASPECT_COLUMNS],
            y=[ASPECT_DESCRIPTIONS[a] for a in ASPECT_COLUMNS],
            title='Ko-okurensi Aspek Non-Neutral'
        )
        fig.update_layout(
            font=dict(family='Inter'),
            title_font=dict(size=16, color='#312e81'),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">☁️ Analisis Kata</div>', unsafe_allow_html=True)

        # Top words per aspect sentiment
        selected_aspect = st.selectbox(
            "Pilih aspek:",
            ASPECT_COLUMNS,
            format_func=lambda x: ASPECT_DESCRIPTIONS[x]
        )
        selected_sentiment = st.selectbox(
            "Pilih sentimen:",
            ['positive', 'negative']
        )

        filtered = train_df[train_df[selected_aspect] == selected_sentiment]

        if len(filtered) > 0:
            from collections import Counter
            all_words = ' '.join(filtered['sentence'].astype(str)).lower().split()
            # Remove common stopwords
            stopwords = {'yang', 'dan', 'di', 'ini', 'itu', 'nya', 'untuk', 'saya',
                        'dengan', 'dari', 'ke', 'tidak', 'juga', 'sudah', 'ada',
                        'bisa', 'lebih', 'kalau', 'kalo', 'tapi', 'tetapi', 'karena',
                        'punya', 'pakai', 'sih', 'aja', 'ya', 'gue', 'gw', 'gua',
                        'kok', 'loh', 'deh', 'dong', 'banget', 'kan', 'masa', 'sama'}
            words = [w for w in all_words if w not in stopwords and len(w) > 2]
            word_counts = Counter(words).most_common(20)

            word_df = pd.DataFrame(word_counts, columns=['Kata', 'Frekuensi'])

            fig = px.bar(
                word_df, x='Frekuensi', y='Kata', orientation='h',
                color='Frekuensi',
                color_continuous_scale='Viridis',
                title=f'Top 20 Kata — {ASPECT_DESCRIPTIONS[selected_aspect]} ({selected_sentiment.capitalize()})'
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                title_font=dict(size=14, color='#312e81'),
                height=500,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Tidak ada data untuk aspek **{ASPECT_DESCRIPTIONS[selected_aspect]}** dengan sentimen **{selected_sentiment}**.")


# ════════════════════════════════════════════════════════
#  PAGE: MULTILABEL CLASSIFICATION
# ════════════════════════════════════════════════════════
elif page == "🏷️ Multilabel Classification":
    st.markdown("""
    <div class="hero-container" style="padding: 1.5rem;">
        <h1 style="font-size:1.8rem;">🏷️ Multilabel Text Classification</h1>
        <p>Training dan evaluasi model klasifikasi multilabel ABSA</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        train_df, valid_df, test_df = load_data()
    except FileNotFoundError:
        st.error("⚠️ Dataset tidak ditemukan.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["⚙️ Konfigurasi & Training", "📈 Hasil Evaluasi", "🔬 Analisis Detail"])

    with tab1:
        st.markdown('<div class="section-header">⚙️ Konfigurasi Model</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📐 Feature Extraction")
            feature_method = st.selectbox(
                "Metode Ekstraksi Fitur:",
                ["TF-IDF", "Count Vectorizer", "TF-IDF + N-gram"]
            )

            if feature_method == "TF-IDF":
                max_features = st.slider("Max Features:", 500, 10000, 3000, 500)
            elif feature_method == "TF-IDF + N-gram":
                max_features = st.slider("Max Features:", 500, 10000, 5000, 500)
                ngram_range = st.selectbox("N-gram Range:", ["(1,1)", "(1,2)", "(1,3)"])
            else:
                max_features = st.slider("Max Features:", 500, 10000, 3000, 500)

        with col2:
            st.markdown("#### 🤖 Model Selection")
            model_type = st.selectbox(
                "Algoritma Klasifikasi:",
                ["Support Vector Machine (SVM)", "Random Forest", "Logistic Regression", "Naive Bayes"]
            )

            multilabel_strategy = st.selectbox(
                "Strategi Multilabel:",
                ["Binary Relevance", "Classifier Chain", "Label Powerset"]
            )

        st.markdown("---")

        # Training button
        if st.button("🚀 Mulai Training", type="primary", use_container_width=True):
            with st.spinner("⏳ Sedang melatih model..."):
                import time
                from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
                from sklearn.svm import SVC, LinearSVC
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.naive_bayes import MultinomialNB
                from sklearn.multiclass import OneVsRestClassifier
                from sklearn.metrics import (
                    classification_report, accuracy_score,
                    f1_score, precision_score, recall_score,
                    hamming_loss
                )
                from sklearn.preprocessing import LabelBinarizer

                progress_bar = st.progress(0, text="Mempersiapkan data...")

                # Feature extraction
                if feature_method == "TF-IDF":
                    vectorizer = TfidfVectorizer(max_features=max_features)
                elif feature_method == "TF-IDF + N-gram":
                    ngram = eval(ngram_range)
                    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram)
                else:
                    vectorizer = CountVectorizer(max_features=max_features)

                X_train = vectorizer.fit_transform(train_df['sentence'].astype(str))
                X_valid = vectorizer.transform(valid_df['sentence'].astype(str))
                X_test = vectorizer.transform(test_df['sentence'].astype(str))

                progress_bar.progress(20, text="Fitur berhasil diekstrak...")

                # Train per aspect
                results = {}
                all_y_true = []
                all_y_pred = []

                for i, aspect in enumerate(ASPECT_COLUMNS):
                    progress_pct = 20 + int((i + 1) / len(ASPECT_COLUMNS) * 60)
                    progress_bar.progress(progress_pct, text=f"Training model untuk aspek: {ASPECT_DESCRIPTIONS[aspect]}...")

                    y_train = train_df[aspect]
                    y_test = test_df[aspect]

                    # Choose model
                    if model_type == "Support Vector Machine (SVM)":
                        clf = LinearSVC(max_iter=10000, random_state=42)
                    elif model_type == "Random Forest":
                        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    elif model_type == "Logistic Regression":
                        clf = LogisticRegression(max_iter=1000, random_state=42)
                    else:
                        if feature_method.startswith("TF-IDF"):
                            from sklearn.naive_bayes import ComplementNB
                            clf = ComplementNB()
                        else:
                            clf = MultinomialNB()

                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    results[aspect] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'report': report,
                        'y_true': y_test.tolist(),
                        'y_pred': y_pred.tolist(),
                    }

                progress_bar.progress(90, text="Menghitung metrik keseluruhan...")
                time.sleep(0.5)
                progress_bar.progress(100, text="✅ Training selesai!")
                time.sleep(0.3)
                progress_bar.empty()

                # Store in session state
                st.session_state['classification_results'] = results
                st.session_state['vectorizer'] = vectorizer
                st.session_state['model_type'] = model_type
                st.session_state['feature_method'] = feature_method

                st.success("✅ Model berhasil dilatih! Lihat hasil di tab **Hasil Evaluasi**.")

    with tab2:
        st.markdown('<div class="section-header">📈 Hasil Evaluasi Model</div>', unsafe_allow_html=True)

        if 'classification_results' not in st.session_state:
            st.info("⚡ Belum ada model yang dilatih. Silakan konfigurasi dan latih model di tab **Konfigurasi & Training**.")
        else:
            results = st.session_state['classification_results']

            # Overview metrics
            st.markdown(f"**Model:** {st.session_state.get('model_type', 'N/A')} | **Fitur:** {st.session_state.get('feature_method', 'N/A')}")

            import plotly.graph_objects as go

            # Metrics per aspect
            aspect_names = [ASPECT_DESCRIPTIONS[a] for a in ASPECT_COLUMNS]
            accuracies = [results[a]['accuracy'] for a in ASPECT_COLUMNS]
            f1_macros = [results[a]['f1_macro'] for a in ASPECT_COLUMNS]
            f1_weights = [results[a]['f1_weighted'] for a in ASPECT_COLUMNS]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Accuracy', x=aspect_names, y=accuracies,
                                marker_color='#667eea'))
            fig.add_trace(go.Bar(name='F1 Macro', x=aspect_names, y=f1_macros,
                                marker_color='#764ba2'))
            fig.add_trace(go.Bar(name='F1 Weighted', x=aspect_names, y=f1_weights,
                                marker_color='#f093fb'))
            fig.update_layout(
                barmode='group',
                title='Perbandingan Metrik per Aspek',
                yaxis=dict(range=[0, 1], title='Skor'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                title_font=dict(size=16, color='#312e81'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            summary_data = []
            for aspect in ASPECT_COLUMNS:
                r = results[aspect]
                summary_data.append({
                    'Aspek': ASPECT_DESCRIPTIONS[aspect],
                    'Accuracy': f"{r['accuracy']:.4f}",
                    'F1 Macro': f"{r['f1_macro']:.4f}",
                    'F1 Weighted': f"{r['f1_weighted']:.4f}",
                    'Precision': f"{r['precision']:.4f}",
                    'Recall': f"{r['recall']:.4f}",
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Average metrics
            avg_acc = np.mean(accuracies)
            avg_f1_macro = np.mean(f1_macros)
            avg_f1_weighted = np.mean(f1_weights)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rata-rata Accuracy", f"{avg_acc:.4f}")
            with col2:
                st.metric("Rata-rata F1 Macro", f"{avg_f1_macro:.4f}")
            with col3:
                st.metric("Rata-rata F1 Weighted", f"{avg_f1_weighted:.4f}")

    with tab3:
        st.markdown('<div class="section-header">🔬 Analisis Detail per Aspek</div>', unsafe_allow_html=True)

        if 'classification_results' not in st.session_state:
            st.info("⚡ Belum ada model yang dilatih.")
        else:
            results = st.session_state['classification_results']

            selected_aspect_detail = st.selectbox(
                "Pilih aspek untuk analisis detail:",
                ASPECT_COLUMNS,
                format_func=lambda x: ASPECT_DESCRIPTIONS[x],
                key="detail_aspect"
            )

            report = results[selected_aspect_detail]['report']
            y_true = results[selected_aspect_detail]['y_true']
            y_pred = results[selected_aspect_detail]['y_pred']

            # Classification report
            st.markdown("#### 📋 Classification Report")
            report_data = []
            for label in SENTIMENT_LABELS:
                if label in report:
                    report_data.append({
                        'Label': label.capitalize(),
                        'Precision': f"{report[label]['precision']:.4f}",
                        'Recall': f"{report[label]['recall']:.4f}",
                        'F1-Score': f"{report[label]['f1-score']:.4f}",
                        'Support': int(report[label]['support']),
                    })
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            # Confusion matrix
            st.markdown("#### 🔲 Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            import plotly.figure_factory as ff

            labels = sorted(set(y_true + y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            fig = px.imshow(
                cm,
                x=labels, y=labels,
                color_continuous_scale='Blues',
                text_auto=True,
                title=f'Confusion Matrix — {ASPECT_DESCRIPTIONS[selected_aspect_detail]}'
            )
            fig.update_layout(
                xaxis_title='Predicted',
                yaxis_title='Actual',
                font=dict(family='Inter'),
                title_font=dict(size=14, color='#312e81'),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
#  PAGE: NAMED ENTITY RECOGNITION
# ════════════════════════════════════════════════════════
elif page == "📛 Named Entity Recognition":
    st.markdown("""
    <div class="hero-container" style="padding: 1.5rem;">
        <h1 style="font-size:1.8rem;">📛 Named Entity Recognition</h1>
        <p>Identifikasi entitas bernama dalam review kendaraan</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── NER Engine ───
    # Known entities database
    BRAND_ENTITIES = {
        'toyota', 'honda', 'daihatsu', 'suzuki', 'mitsubishi', 'nissan',
        'yamaha', 'kawasaki', 'wuling', 'mazda', 'bmw', 'isuzu', 'hyundai',
        'kia', 'mercedes', 'mini', 'esemka', 'ford', 'chevrolet'
    }

    PRODUCT_ENTITIES = {
        'avanza', 'xenia', 'brio', 'jazz', 'hrv', 'crv', 'mobilio', 'calya',
        'sigra', 'agya', 'ayla', 'innova', 'fortuner', 'rush', 'terios',
        'xpander', 'pajero', 'ertiga', 'ignis', 'sirion', 'vario', 'beat',
        'scoopy', 'nmax', 'aerox', 'mio', 'vios', 'altis', 'camry', 'yaris',
        'confero', 'cortez', 'almaz', 'formo', 'livina', 'freed', 'lexi',
        'veloz', 'pcx', 'sonic', 'cbr', 'karimun', 'carry', 'alphard',
        'panther', 'taruna', 'spacy', 'colt', 'satria', 'athlete', 'revo',
        'xride', 'prado', 'grandmax', 'splash', 'wagon', 'l300'
    }

    LOCATION_ENTITIES = {
        'jakarta', 'bandung', 'surabaya', 'bali', 'jogja', 'yogyakarta',
        'semarang', 'medan', 'makassar', 'palembang', 'malang', 'solo',
        'denpasar', 'poso', 'tasikmalaya', 'nagrek', 'dieng', 'sumatra',
        'jawa', 'kalimantan', 'sulawesi', 'cihampelas', 'balaraja',
        'sisingamangaraja', 'pondok indah', 'cikampek', 'indonesia',
        'sukabumi', 'timur'
    }

    ASPECT_TERM_ENTITIES = {
        'bbm', 'bensin', 'solar', 'bahan bakar', 'konsumsi',
        'mesin', 'tenaga', 'tarikan', 'akselerasi', 'power', 'torsi',
        'interior', 'eksterior', 'desain', 'kabin', 'jok', 'dasbor',
        'dashboard', 'suspensi', 'kaki-kaki', 'rem', 'kopling', 'setir',
        'ac', 'lampu', 'spion', 'bagasi', 'airbag', 'fitur', 'sparepart',
        'onderdil', 'suku cadang', 'ban', 'knalpot', 'radiator',
        'harga', 'pajak', 'purnajual', 'murah', 'mahal', 'irit', 'boros',
        'servis', 'service', 'bengkel', 'dealer', 'ahass', 'ahas',
        'perawatan', 'inden', 'booking', 'teknisi', 'mekanik', 'montir'
    }

    def extract_entities(text):
        """Extract named entities from text using rule-based approach."""
        text_lower = text.lower()
        words = text_lower.split()
        entities = []

        # Brand detection
        for brand in BRAND_ENTITIES:
            if brand in text_lower:
                start = text_lower.find(brand)
                original = text[start:start+len(brand)]
                entities.append({
                    'text': original,
                    'label': 'BRAND',
                    'start': start,
                    'end': start + len(brand),
                    'color': '#dbeafe',
                    'text_color': '#1e40af',
                    'border_color': '#93c5fd'
                })

        # Product detection
        for product in PRODUCT_ENTITIES:
            if product in text_lower:
                start = text_lower.find(product)
                original = text[start:start+len(product)]
                entities.append({
                    'text': original,
                    'label': 'PRODUCT',
                    'start': start,
                    'end': start + len(product),
                    'color': '#dcfce7',
                    'text_color': '#166534',
                    'border_color': '#86efac'
                })

        # Location detection
        for location in LOCATION_ENTITIES:
            if location in text_lower:
                start = text_lower.find(location)
                original = text[start:start+len(location)]
                entities.append({
                    'text': original,
                    'label': 'LOCATION',
                    'start': start,
                    'end': start + len(location),
                    'color': '#fef3c7',
                    'text_color': '#92400e',
                    'border_color': '#fcd34d'
                })

        # Aspect term detection
        for term in ASPECT_TERM_ENTITIES:
            if term in text_lower:
                start = text_lower.find(term)
                original = text[start:start+len(term)]
                entities.append({
                    'text': original,
                    'label': 'ASPECT',
                    'start': start,
                    'end': start + len(term),
                    'color': '#f3e8ff',
                    'text_color': '#6b21a8',
                    'border_color': '#c084fc'
                })

        # Sort by start position and remove overlaps
        entities.sort(key=lambda x: x['start'])
        filtered = []
        last_end = -1
        for ent in entities:
            if ent['start'] >= last_end:
                filtered.append(ent)
                last_end = ent['end']

        return filtered

    def render_ner_html(text, entities):
        """Render text with highlighted entities."""
        if not entities:
            return f'<p style="font-size:1rem; line-height:1.8;">{text}</p>'

        html_parts = []
        last_end = 0

        for ent in entities:
            # Add text before entity
            if ent['start'] > last_end:
                html_parts.append(text[last_end:ent['start']])

            # Add highlighted entity
            html_parts.append(
                f'<span style="background:{ent["color"]}; color:{ent["text_color"]}; '
                f'border:1px solid {ent["border_color"]}; padding:2px 6px; '
                f'border-radius:4px; font-weight:600; margin:0 1px;">'
                f'{text[ent["start"]:ent["end"]]}'
                f'<sup style="font-size:0.6rem; margin-left:2px; opacity:0.8;">{ent["label"]}</sup>'
                f'</span>'
            )
            last_end = ent['end']

        # Add remaining text
        if last_end < len(text):
            html_parts.append(text[last_end:])

        return f'<p style="font-size:1rem; line-height:2.2;">{"".join(html_parts)}</p>'

    # ─── NER Interface ───
    tab1, tab2 = st.tabs(["🔍 NER Interaktif", "📊 Statistik Entitas"])

    with tab1:
        st.markdown('<div class="section-header">🔍 Ekstraksi Entitas</div>', unsafe_allow_html=True)

        # Entity legend
        st.markdown("""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:1rem;">
            <span class="ner-entity ner-brand">🏭 BRAND</span>
            <span class="ner-entity ner-product">🚙 PRODUCT</span>
            <span class="ner-entity ner-location">📍 LOCATION</span>
            <span class="ner-entity ner-aspect">🏷️ ASPECT</span>
        </div>
        """, unsafe_allow_html=True)

        ner_input = st.text_area(
            "Masukkan teks review:",
            value="Saya pakai Toyota Avanza di Jakarta, mesin nya bandel dan irit bensin. Servis di bengkel resmi juga murah.",
            height=120,
            key="ner_input"
        )

        if st.button("🔍 Ekstrak Entitas", type="primary", key="ner_btn"):
            entities = extract_entities(ner_input)
            ner_html = render_ner_html(ner_input, entities)

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("#### 📄 Hasil NER:")
            st.markdown(ner_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if entities:
                st.markdown("#### 📋 Entitas Terdeteksi:")
                ent_data = []
                for ent in entities:
                    ent_data.append({
                        'Teks': ent['text'],
                        'Label': ent['label'],
                        'Posisi Awal': ent['start'],
                        'Posisi Akhir': ent['end'],
                    })
                ent_df = pd.DataFrame(ent_data)
                st.dataframe(ent_df, use_container_width=True, hide_index=True)

        # Sample reviews NER
        st.markdown("---")
        st.markdown("#### 📝 Contoh Review dari Dataset:")

        try:
            train_df, _, _ = load_data()
            sample_reviews = train_df['sentence'].sample(5, random_state=42).tolist()
        except Exception:
            sample_reviews = [
                "Toyota Avanza irit banget bensin nya",
                "Honda Brio mesin nya halus dan bagus",
                "Servis di bengkel Daihatsu Sukabumi bagus",
            ]

        for review in sample_reviews:
            entities = extract_entities(review)
            ner_html = render_ner_html(review, entities)
            st.markdown(f"""
            <div class="prediction-card">
                {ner_html}
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">📊 Statistik Entitas dalam Dataset</div>', unsafe_allow_html=True)

        try:
            train_df, valid_df, test_df = load_data()
            all_data = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        except Exception:
            st.error("Dataset tidak ditemukan.")
            st.stop()

        with st.spinner("Menganalisis entitas dalam seluruh dataset..."):
            from collections import Counter
            brand_counter = Counter()
            product_counter = Counter()
            location_counter = Counter()
            aspect_counter = Counter()
            entity_type_counts = {'BRAND': 0, 'PRODUCT': 0, 'LOCATION': 0, 'ASPECT': 0}

            for text in all_data['sentence'].astype(str):
                entities = extract_entities(text)
                for ent in entities:
                    entity_type_counts[ent['label']] += 1
                    if ent['label'] == 'BRAND':
                        brand_counter[ent['text'].lower()] += 1
                    elif ent['label'] == 'PRODUCT':
                        product_counter[ent['text'].lower()] += 1
                    elif ent['label'] == 'LOCATION':
                        location_counter[ent['text'].lower()] += 1
                    elif ent['label'] == 'ASPECT':
                        aspect_counter[ent['text'].lower()] += 1

        import plotly.express as px

        # Entity type distribution
        type_df = pd.DataFrame([
            {'Tipe Entitas': k, 'Jumlah': v}
            for k, v in entity_type_counts.items()
        ])

        fig = px.pie(
            type_df, values='Jumlah', names='Tipe Entitas',
            color_discrete_sequence=['#667eea', '#43e97b', '#fcd34d', '#c084fc'],
            title='Distribusi Tipe Entitas',
            hole=0.4,
        )
        fig.update_layout(
            font=dict(family='Inter'),
            title_font=dict(size=16, color='#312e81'),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top entities per type
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏭 Top Brand")
            if brand_counter:
                brand_df = pd.DataFrame(brand_counter.most_common(10), columns=['Brand', 'Count'])
                fig = px.bar(brand_df, x='Count', y='Brand', orientation='h',
                            color_discrete_sequence=['#667eea'])
                fig.update_layout(yaxis=dict(autorange="reversed"), height=350,
                                plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📍 Top Location")
            if location_counter:
                loc_df = pd.DataFrame(location_counter.most_common(10), columns=['Location', 'Count'])
                fig = px.bar(loc_df, x='Count', y='Location', orientation='h',
                            color_discrete_sequence=['#fcd34d'])
                fig.update_layout(yaxis=dict(autorange="reversed"), height=350,
                                plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 🚙 Top Product")
            if product_counter:
                prod_df = pd.DataFrame(product_counter.most_common(10), columns=['Product', 'Count'])
                fig = px.bar(prod_df, x='Count', y='Product', orientation='h',
                            color_discrete_sequence=['#43e97b'])
                fig.update_layout(yaxis=dict(autorange="reversed"), height=350,
                                plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🏷️ Top Aspect Terms")
            if aspect_counter:
                asp_df = pd.DataFrame(aspect_counter.most_common(10), columns=['Aspect', 'Count'])
                fig = px.bar(asp_df, x='Count', y='Aspect', orientation='h',
                            color_discrete_sequence=['#c084fc'])
                fig.update_layout(yaxis=dict(autorange="reversed"), height=350,
                                plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'))
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
#  PAGE: DEMO PREDIKSI
# ════════════════════════════════════════════════════════
elif page == "🚀 Demo Prediksi":
    st.markdown("""
    <div class="hero-container" style="padding: 1.5rem;">
        <h1 style="font-size:1.8rem;">🚀 Demo Prediksi ABSA</h1>
        <p>Coba prediksi sentimen aspek & NER pada review kendaraan</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Quick Train if needed ───
    @st.cache_resource
    def train_quick_model():
        """Train a quick model for demo purposes."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC

        try:
            base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
            train_df = pd.read_csv(os.path.join(base_path, 'train_preprocess.csv'))
        except FileNotFoundError:
            return None, None

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(train_df['sentence'].astype(str))

        models = {}
        for aspect in ASPECT_COLUMNS:
            clf = LinearSVC(max_iter=10000, random_state=42)
            clf.fit(X_train, train_df[aspect])
            models[aspect] = clf

        return vectorizer, models

    st.markdown('<div class="section-header">✍️ Masukkan Review</div>', unsafe_allow_html=True)

    demo_input = st.text_area(
        "Tulis review kendaraan di sini:",
        value="",
        height=120,
        placeholder="Contoh: Toyota Avanza irit banget bensin nya dan mesin nya bandel. Servis di bengkel resmi juga murah. Sayang harga nya agak mahal.",
        key="demo_input"
    )

    # Quick examples
    st.markdown("**💡 Atau pilih contoh:**")
    example_cols = st.columns(3)
    examples = [
        "Honda Beat irit banget bahan bakar nya, mesin halus dan tarikan responsif",
        "Bengkel Toyota pelayanan nya jelek banget, lama dan mahal",
        "Xpander desain bagus harga terjangkau tapi mesin kurang bertenaga"
    ]
    for col, example in zip(example_cols, examples):
        with col:
            if st.button(f"📝 {example[:40]}...", key=f"ex_{example[:10]}"):
                st.session_state['demo_input'] = example
                st.rerun()

    if demo_input or st.session_state.get('demo_input'):
        text_to_analyze = demo_input or st.session_state.get('demo_input', '')

        if text_to_analyze.strip():
            st.markdown("---")

            with st.spinner("🔄 Memproses prediksi..."):
                vectorizer, models = train_quick_model()

                if vectorizer is None:
                    st.error("Dataset tidak ditemukan untuk training model demo.")
                    st.stop()

                # Predict
                X_input = vectorizer.transform([text_to_analyze])
                predictions = {}
                for aspect in ASPECT_COLUMNS:
                    predictions[aspect] = models[aspect].predict(X_input)[0]

                # NER
                entities = extract_entities(text_to_analyze)

            # ─── Display Results ───
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown('<div class="section-header">🏷️ Prediksi Sentimen Aspek</div>', unsafe_allow_html=True)

                for aspect in ASPECT_COLUMNS:
                    sentiment = predictions[aspect]
                    icon = ASPECT_DESCRIPTIONS[aspect]

                    if sentiment == 'positive':
                        sent_badge = '<span class="sentiment-positive">✅ Positive</span>'
                        bar_color = '#00b09b'
                    elif sentiment == 'negative':
                        sent_badge = '<span class="sentiment-negative">❌ Negative</span>'
                        bar_color = '#eb3349'
                    else:
                        sent_badge = '<span class="sentiment-neutral">➖ Neutral</span>'
                        bar_color = '#95a5a6'

                    st.markdown(f"""
                    <div class="prediction-card" style="display:flex; justify-content:space-between; align-items:center; padding:0.8rem 1.2rem;">
                        <span style="font-weight:600; font-size:0.95rem;">{icon}</span>
                        {sent_badge}
                    </div>
                    """, unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="section-header">📛 Entitas Terdeteksi (NER)</div>', unsafe_allow_html=True)

                ner_html = render_ner_html(text_to_analyze, entities)
                st.markdown(f"""
                <div class="prediction-card">
                    {ner_html}
                </div>
                """, unsafe_allow_html=True)

                if entities:
                    st.markdown("##### 📋 Detail Entitas:")
                    for ent in entities:
                        label_colors = {
                            'BRAND': ('🏭', '#dbeafe', '#1e40af'),
                            'PRODUCT': ('🚙', '#dcfce7', '#166534'),
                            'LOCATION': ('📍', '#fef3c7', '#92400e'),
                            'ASPECT': ('🏷️', '#f3e8ff', '#6b21a8'),
                        }
                        emoji, bg, tc = label_colors.get(ent['label'], ('', '#f3f4f6', '#333'))
                        st.markdown(f"""
                        <div style="display:flex; align-items:center; gap:8px; margin:4px 0;
                                    padding:6px 10px; background:{bg}; border-radius:8px;">
                            <span>{emoji}</span>
                            <span style="font-weight:600; color:{tc};">{ent['text']}</span>
                            <span style="font-size:0.7rem; opacity:0.7;">({ent['label']})</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Tidak ada entitas terdeteksi.")

            # Summary visualization
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Ringkasan Prediksi</div>', unsafe_allow_html=True)

            import plotly.graph_objects as go

            # Sentiment radar chart
            sentiment_scores = []
            for aspect in ASPECT_COLUMNS:
                s = predictions[aspect]
                if s == 'positive':
                    sentiment_scores.append(1)
                elif s == 'negative':
                    sentiment_scores.append(-1)
                else:
                    sentiment_scores.append(0)

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[abs(s) if s != 0 else 0.1 for s in sentiment_scores],
                theta=[ASPECT_DESCRIPTIONS[a] for a in ASPECT_COLUMNS],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='#667eea', width=2),
                marker=dict(
                    color=['#00b09b' if s > 0 else '#eb3349' if s < 0 else '#95a5a6' for s in sentiment_scores],
                    size=10,
                ),
                text=[predictions[a].capitalize() for a in ASPECT_COLUMNS],
                hovertemplate='%{theta}<br>Sentimen: %{text}<extra></extra>',
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1.2]),
                ),
                showlegend=False,
                title='Radar Chart Sentimen Aspek',
                font=dict(family='Inter'),
                title_font=dict(size=14, color='#312e81'),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────── Footer ───────────────────
st.markdown("""
<div class="footer">
    <p>🔍 ABSA - Multilabel Classification & NER | Pemrosesan Bahasa Alami</p>
    <p>Universitas Atma Jaya • Semester 6 • 2026</p>
</div>
""", unsafe_allow_html=True)
