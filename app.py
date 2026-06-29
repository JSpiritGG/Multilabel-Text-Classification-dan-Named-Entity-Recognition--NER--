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
    .badge-PRODUCT { background: linear-gradient(135deg, #43e97b, #38f9d7); }
    .badge-PRICE { background: linear-gradient(135deg, #a18cd1, #fbc2eb); }
    .badge-PLACE { background: linear-gradient(135deg, #ffecd2, #fcb69f); color: #333; }
    .badge-PROMOTION { background: linear-gradient(135deg, #f093fb, #f5576c); }

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
    base_path = os.path.join(os.path.dirname(__file__), 'dataset')
    train_df = pd.read_csv(os.path.join(base_path, 'Kelp2_multilabel_train.csv'))
    valid_df = pd.read_csv(os.path.join(base_path, 'Kelp2_multilabel_val.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'Kelp2_multilabel_test.csv'))
    return train_df, valid_df, test_df


ASPECT_COLUMNS = ['PRODUCT', 'PRICE', 'PLACE', 'PROMOTION']
SENTIMENT_LABELS = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

ASPECT_DESCRIPTIONS = {
    'PRODUCT': '👕 Produk (Product)',
    'PRICE': '💰 Harga (Price)',
    'PLACE': '🏪 Tempat/Toko (Place)',
    'PROMOTION': '🏷️ Promosi (Promotion)'
}

ASPECT_COLORS = {
    'PRODUCT': '#1e40af',
    'PRICE': '#1e40af',
    'PLACE': '#1e40af',
    'PROMOTION': '#1e40af'
}

import re

@st.cache_resource
def load_ner_model():
    """Load the best NER model from joblib."""
    try:
        import joblib
        base_path = os.path.join(os.path.dirname(__file__), 'models')
        data = joblib.load(os.path.join(base_path, 'Kelp2_best_ner_model.joblib'))
        return data
    except Exception as e:
        return None

def _ner_token_shape(token):
    """Token shape EXACTLY as in training (collapsed a/d/p runs)."""
    parts = []
    for ch in str(token):
        if ch.isdigit():
            c = 'd'
        elif ch.isalpha():
            c = 'a'
        else:
            c = 'p'
        if not parts or parts[-1] != c:
            parts.append(c)
    return ''.join(parts)


def extract_features_for_token(tokens, idx, majority_tag=None):
    """
    EXACT replica of `feature_token` from Kelp2_ner_modeling.ipynb (cell 56).
    The DictVectorizer in the saved payload was fitted on these exact feature
    keys, so any deviation (different key names, missing gazetteer, etc.) makes
    transform() silently drop the columns to zero and breaks inference.
    `majority_tag` is the gazetteer dict saved inside the joblib payload.
    """
    if majority_tag is None:
        majority_tag = {}

    token = str(tokens[idx]).lower()
    n = len(tokens)
    feats = {
        'bias': 1,
        'tok=' + token: 1,
        'majority=' + majority_tag.get(token, 'UNK'): 1,
        'prefix1=' + token[:1]: 1,
        'prefix2=' + token[:2]: 1,
        'prefix3=' + token[:3]: 1,
        'prefix4=' + token[:4]: 1,
        'suffix1=' + token[-1:]: 1,
        'suffix2=' + token[-2:]: 1,
        'suffix3=' + token[-3:]: 1,
        'suffix4=' + token[-4:]: 1,
        'shape=' + _ner_token_shape(token): 1,
        'len=' + str(min(len(token), 12)): 1,
        'has_digit=' + str(any(ch.isdigit() for ch in token)): 1,
        'pos_bin=' + str(int(idx / max(1, n - 1) * 4)): 1,
    }
    for off in [-3, -2, -1, 1, 2, 3]:
        j = idx + off
        if 0 <= j < n:
            ctx = str(tokens[j]).lower()
            feats[f'w{off}=' + ctx] = 1
            feats[f'majority{off}=' + majority_tag.get(ctx, 'UNK')] = 1
        else:
            feats[f'edge{off}'] = 1
    if idx > 0:
        feats['prev_cur=' + str(tokens[idx-1]).lower() + '_' + token] = 1
    if idx < n - 1:
        feats['cur_next=' + token + '_' + str(tokens[idx+1]).lower()] = 1
    if 0 < idx < n - 1:
        feats['tri=' + str(tokens[idx-1]).lower() + '_' + token + '_' + str(tokens[idx+1]).lower()] = 1
    return feats


def bio_to_entities(tokens, tags, original_text, confidences=None):
    """Convert BIO token tags back to entity spans in original text.

    `confidences` (optional): per-token max-probability array. When present,
    each entity gets a 'confidence' field = mean confidence over its tokens.
    """
    entities = []
    i = 0
    # Reconstruct char offsets by scanning original_text
    char_pos = 0
    token_offsets = []
    for tok in tokens:
        start = original_text.lower().find(tok.lower(), char_pos)
        if start == -1:
            start = char_pos
        end = start + len(tok)
        token_offsets.append((start, end))
        char_pos = end

    def conf_at(idx):
        if confidences is None:
            return None
        try:
            return float(confidences[idx])
        except Exception:
            return None

    current_entity = None
    current_confs = []
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith('B-'):
            if current_entity:
                current_entity['confidence'] = (
                    sum(current_confs) / len(current_confs) if current_confs else None
                )
                entities.append(current_entity)
            label = tag[2:]  # full label like PRODUCT_NEGATIVE, PLACE_POSITIVE, etc.
            display_label = _bio_label_to_display(label)
            start, end = token_offsets[i]
            current_entity = {
                'start': start,
                'end': end,
                'label': display_label,
                'text': original_text[start:end],
                'confidence': None,
            }
            current_confs = [c for c in [conf_at(i)] if c is not None]
        elif tag.startswith('I-') and current_entity:
            start, end = token_offsets[i]
            current_entity['end'] = end
            current_entity['text'] = original_text[current_entity['start']:end]
            c = conf_at(i)
            if c is not None:
                current_confs.append(c)
        else:
            if current_entity:
                current_entity['confidence'] = (
                    sum(current_confs) / len(current_confs) if current_confs else None
                )
                entities.append(current_entity)
                current_entity = None
                current_confs = []
    if current_entity:
        current_entity['confidence'] = (
            sum(current_confs) / len(current_confs) if current_confs else None
        )
        entities.append(current_entity)
    return entities

def _bio_label_to_display(label):
    """Map BIO label (e.g. PRODUCT_NEGATIVE) to display category."""
    label_upper = label.upper()
    if 'PRODUCT' in label_upper:
        return 'PRODUCT'
    elif 'PLACE' in label_upper or 'LOCATION' in label_upper:
        return 'LOCATION'
    elif 'PRICE' in label_upper:
        return 'PRICE'
    elif 'PROMOTION' in label_upper or 'SERVICE' in label_upper:
        return 'SERVICE'
    elif 'BRAND' in label_upper:
        return 'BRAND'
    else:
        return 'ASPECT'

def extract_entities(text):
    """Extract entities using the best NER ML model (SGD + DictVectorizer)."""
    ner_data = load_ner_model()

    # Fallback to rule-based if model not available
    if ner_data is None:
        return _extract_entities_fallback(text)

    try:
        classifier = ner_data.get('classifier')
        vectorizer = ner_data.get('vectorizer')  # DictVectorizer
        majority_tag = ner_data.get('majority_tag', {})  # gazetteer from training
        if classifier is None or vectorizer is None:
            return _extract_entities_fallback(text)

        tokens = text.split()
        if not tokens:
            return []

        features = [extract_features_for_token(tokens, i, majority_tag) for i in range(len(tokens))]
        X = vectorizer.transform(features)
        tags = classifier.predict(X)

        # Confidence per token (hanya jika model mendukung probabilitas, mis. loss='log_loss')
        confidences = None
        if ner_data.get('supports_proba') and hasattr(classifier, 'predict_proba'):
            try:
                proba = classifier.predict_proba(X)
                confidences = proba.max(axis=1)
            except Exception:
                confidences = None

        entities = bio_to_entities(tokens, tags, text, confidences)
        return sorted(entities, key=lambda x: x['start'])

    except Exception:
        return _extract_entities_fallback(text)

def _extract_entities_fallback(text):
    """Fallback rule-based NER in case model is unavailable."""
    BRAND_ENTITIES = {
        'zano', 'hos of shopaholic', 'hos', 'shopaholic', 'uniqlo', 'zara',
        'h&m', 'matahari', 'ramayana', 'borma', 'erigo', '3second', 'greenlight',
        'levis', 'adidas', 'nike', 'puma', 'champion'
    }
    PRODUCT_ENTITIES = {
        'baju', 'kaos', 'kemeja', 'celana', 'rok', 'jaket', 'sweater', 'hoodie',
        'dress', 'gamis', 'tunik', 'blouse', 'jas', 'pakaian', 'outfit', 'jeans',
        'chino', 'kulot', 'legging', 'kain', 'bahan'
    }
    LOCATION_ENTITIES = {
        'jakarta', 'bandung', 'surabaya', 'bali', 'jogja', 'yogyakarta', 'semarang',
        'medan', 'makassar', 'palembang', 'malang', 'solo', 'denpasar', 'tasikmalaya',
        'mall', 'plaza', 'toko', 'butik', 'outlet', 'store', 'cabang'
    }
    ASPECT_TERM_ENTITIES = {
        'harga', 'murah', 'mahal', 'diskon', 'promo', 'promosi', 'sale',
        'kualitas', 'bagus', 'jelek', 'awet', 'luntur', 'sobek', 'jahitan',
        'pelayanan', 'ramah', 'kasir', 'pramuniaga', 'mbak', 'mas', 'layan',
        'tempat', 'bersih', 'kotor', 'luas', 'sempit', 'nyaman', 'parkir',
        'fitting room', 'kamar ganti', 'koleksi', 'lengkap', 'ukuran', 'size'
    }
    text_lower = text.lower()
    entities = []

    def add_entity(entity_type, word):
        for match in re.finditer(rf'\b{re.escape(word)}\b', text_lower):
            start, end = match.start(), match.end()
            overlap = any(start < e['end'] and end > e['start'] for e in entities)
            if not overlap:
                entities.append({'start': start, 'end': end, 'label': entity_type, 'text': text[start:end]})

    for word in sorted(BRAND_ENTITIES, key=len, reverse=True): add_entity('BRAND', word)
    for word in sorted(PRODUCT_ENTITIES, key=len, reverse=True): add_entity('PRODUCT', word)
    for word in sorted(LOCATION_ENTITIES, key=len, reverse=True): add_entity('LOCATION', word)
    for word in sorted(ASPECT_TERM_ENTITIES, key=len, reverse=True): add_entity('ASPECT', word)
    return sorted(entities, key=lambda x: x['start'])

def render_ner_html(text, entities):
    if not entities:
        return f'<p style="font-size:1rem; line-height:2.2;">{text}</p>'
        
    html_parts = []
    last_end = 0
    
    for ent in entities:
        if ent['label'] == 'BRAND':
            bg_color = '#dbeafe'
            border_color = '#93c5fd'
            text_color = '#1e40af'
        elif ent['label'] == 'PRODUCT':
            bg_color = '#dcfce7'
            border_color = '#86efac'
            text_color = '#166534'
        elif ent['label'] == 'LOCATION':
            bg_color = '#fef3c7'
            border_color = '#fcd34d'
            text_color = '#92400e'
        elif ent['label'] == 'ASPECT':
            bg_color = '#f3e8ff'
            border_color = '#c084fc'
            text_color = '#6b21a8'
        else:
            bg_color = '#f3f4f6'
            border_color = '#e5e7eb'
            text_color = '#374151'
            
        # Add text before entity
        if ent['start'] > last_end:
            html_parts.append(text[last_end:ent['start']])
            
        # Add highlighted entity
        conf = ent.get('confidence')
        conf_title = f' title="confidence: {conf*100:.1f}%"' if conf is not None else ''
        conf_sup = f' · {conf*100:.0f}%' if conf is not None else ''
        html_parts.append(
            f'<span{conf_title} style="background:{bg_color}; color:{text_color}; '
            f'border:1px solid {border_color}; padding:2px 6px; '
            f'border-radius:4px; font-weight:600; margin:0 1px;">'
            f'{text[ent["start"]:ent["end"]]}'
            f'<sup style="font-size:0.6rem; margin-left:2px; opacity:0.8;">{ent["label"]}{conf_sup}</sup>'
            f'</span>'
        )
        last_end = ent['end']
        
    # Add remaining text
    if last_end < len(text):
        html_parts.append(text[last_end:])
        
    return f'<p style="font-size:1rem; line-height:2.2;">{"".join(html_parts)}</p>'


# ─────────────────── Sidebar ───────────────────
with st.sidebar:
    st.markdown("## 🔍 ABSA Navigator")
    st.markdown("---")

    page = st.radio(
        "📑 Pilih Halaman",
        [
            "🏷️ Multilabel Classification",
            "📛 Named Entity Recognition",
            "🚀 Demo Prediksi",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; opacity: 0.7; font-size: 0.75rem;'>
        <p>🔍 ABSA Dashboard</p>
        <p>Multilabel Classification & NER</p>
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

        train_df['word_count'] = train_df['text'].apply(lambda x: len(str(x).split()))

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
            for sentiment in SENTIMENT_LABELS:
                col_name = f"{aspect}_{sentiment}"
                count = train_df[col_name].sum() if col_name in train_df.columns else 0
                sentiment_data.append({
                    'Aspek': ASPECT_DESCRIPTIONS.get(aspect, aspect),
                    'Sentimen': sentiment.capitalize(),
                    'Jumlah': count
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
            pos_col = f"{aspect}_POSITIVE"
            neg_col = f"{aspect}_NEGATIVE"
            neu_col = f"{aspect}_NEUTRAL"
            
            pos_count = train_df[pos_col].sum() if pos_col in train_df.columns else 0
            neg_count = train_df[neg_col].sum() if neg_col in train_df.columns else 0
            neu_count = train_df[neu_col].sum() if neu_col in train_df.columns else 0
            
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
            pos_col = f"{aspect}_POSITIVE"
            neg_col = f"{aspect}_NEGATIVE"
            
            has_pos = train_df[pos_col] if pos_col in train_df.columns else 0
            has_neg = train_df[neg_col] if neg_col in train_df.columns else 0
            binary_df[aspect] = (has_pos | has_neg).astype(int)

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
            ['POSITIVE', 'NEGATIVE']
        )
        
        target_col = f"{selected_aspect}_{selected_sentiment}"
        if target_col in train_df.columns:
            filtered = train_df[train_df[target_col] == 1]
        else:
            filtered = pd.DataFrame()

        if len(filtered) > 0:
            from collections import Counter
            all_words = ' '.join(filtered['text'].astype(str)).lower().split()
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
            st.info(f"Tidak ada data untuk aspek **{ASPECT_DESCRIPTIONS[selected_aspect]}** dengan sentimen **{selected_sentiment.capitalize()}**.")


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

                X_train = vectorizer.fit_transform(train_df['text'].astype(str))
                X_valid = vectorizer.transform(valid_df['text'].astype(str))
                X_test = vectorizer.transform(test_df['text'].astype(str))

                progress_bar.progress(20, text="Fitur berhasil diekstrak...")

                # Define target columns
                target_cols = []
                for a in ASPECT_COLUMNS:
                    for s in SENTIMENT_LABELS:
                        col = f"{a}_{s}"
                        if col in train_df.columns:
                            target_cols.append(col)

                y_train = train_df[target_cols].values
                y_test = test_df[target_cols].values

                progress_bar.progress(30, text=f"Melatih model dengan {multilabel_strategy}...")
                
                # Gunakan sklearn-native multilabel strategies (tanpa library tambahan)
                from sklearn.multiclass import OneVsRestClassifier
                from sklearn.multioutput import ClassifierChain as SklearnChain

                # Choose base model
                if model_type == "Support Vector Machine (SVM)":
                    base_clf = LinearSVC(max_iter=10000, random_state=42, class_weight='balanced')
                elif model_type == "Random Forest":
                    base_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
                elif model_type == "Logistic Regression":
                    base_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
                else:
                    base_clf = MultinomialNB()

                if multilabel_strategy == "Binary Relevance":
                    clf = OneVsRestClassifier(base_clf)
                elif multilabel_strategy == "Classifier Chain":
                    clf = SklearnChain(base_clf, order='random', random_state=42)
                else:
                    # Label Powerset tidak ada di sklearn, fallback ke OneVsRestClassifier
                    clf = OneVsRestClassifier(base_clf)
                    st.info("ℹ️ Label Powerset tidak tersedia di scikit-learn standar. Menggunakan Binary Relevance (OneVsRestClassifier) sebagai pengganti.")

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Convert sparse matrix to dense array if needed
                if hasattr(y_pred, 'toarray'):
                    y_pred_array = y_pred.toarray()
                else:
                    y_pred_array = np.array(y_pred)

                progress_bar.progress(90, text="Menghitung metrik evaluasi...")
                
                report = classification_report(y_test, y_pred_array, target_names=target_cols, output_dict=True, zero_division=0)
                
                results = {
                    'accuracy': accuracy_score(y_test, y_pred_array),
                    'f1_macro': f1_score(y_test, y_pred_array, average='macro', zero_division=0),
                    'f1_weighted': f1_score(y_test, y_pred_array, average='weighted', zero_division=0),
                    'precision': precision_score(y_test, y_pred_array, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred_array, average='weighted', zero_division=0),
                    'hamming_loss': hamming_loss(y_test, y_pred_array),
                    'report': report,
                    'y_true': y_test.tolist(),
                    'y_pred': y_pred_array.tolist(),
                    'target_cols': target_cols,
                    'model': clf
                }

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
            target_cols = results['target_cols']
            report = results['report']

            # Overview metrics
            st.markdown(f"**Model:** {st.session_state.get('model_type', 'N/A')} | **Fitur:** {st.session_state.get('feature_method', 'N/A')}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Exact Match Accuracy", f"{results['accuracy']:.4f}")
            with col2:
                st.metric("F1 Macro", f"{results['f1_macro']:.4f}")
            with col3:
                st.metric("F1 Weighted", f"{results['f1_weighted']:.4f}")
            with col4:
                st.metric("Hamming Loss", f"{results['hamming_loss']:.4f}")

            import plotly.graph_objects as go

            # Metrics per target column
            f1_scores = [report[col]['f1-score'] for col in target_cols]
            precisions = [report[col]['precision'] for col in target_cols]
            recalls = [report[col]['recall'] for col in target_cols]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='F1-Score', x=target_cols, y=f1_scores, marker_color='#667eea'))
            fig.add_trace(go.Bar(name='Precision', x=target_cols, y=precisions, marker_color='#764ba2'))
            fig.add_trace(go.Bar(name='Recall', x=target_cols, y=recalls, marker_color='#f093fb'))
            
            fig.update_layout(
                barmode='group',
                title='Metrik per Target Aspek-Sentimen',
                yaxis=dict(range=[0, 1], title='Skor'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                title_font=dict(size=16, color='#312e81'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            summary_data = []
            for col in target_cols:
                summary_data.append({
                    'Target': col,
                    'Precision': f"{report[col]['precision']:.4f}",
                    'Recall': f"{report[col]['recall']:.4f}",
                    'F1-Score': f"{report[col]['f1-score']:.4f}",
                    'Support': int(report[col]['support'])
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown('<div class="section-header">🔬 Analisis Detail per Target</div>', unsafe_allow_html=True)

        if 'classification_results' not in st.session_state:
            st.info("⚡ Belum ada model yang dilatih.")
        else:
            results = st.session_state['classification_results']
            target_cols = results['target_cols']

            selected_target = st.selectbox(
                "Pilih target aspek-sentimen untuk analisis detail:",
                target_cols,
                key="detail_target"
            )

            # Confusion matrix
            st.markdown("#### 🔲 Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            import plotly.express as px

            target_idx = target_cols.index(selected_target)
            y_true_col = [row[target_idx] for row in results['y_true']]
            y_pred_col = [row[target_idx] for row in results['y_pred']]

            labels = [0, 1]
            cm = confusion_matrix(y_true_col, y_pred_col, labels=labels)

            fig = px.imshow(
                cm,
                x=['Predicted 0', 'Predicted 1'], 
                y=['Actual 0', 'Actual 1'],
                color_continuous_scale='Blues',
                text_auto=True,
                title=f'Confusion Matrix — {selected_target}'
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

    # NER uses global extract_entities() and render_ner_html() functions defined above

    # ─── NER Interface ───
    tab1, tab2 = st.tabs(["🔍 NER Interaktif", "📊 Statistik Entitas"])

    with tab1:
        st.markdown('<div class="section-header">🔍 Ekstraksi Entitas</div>', unsafe_allow_html=True)

        # Entity legend
        st.markdown("""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:1rem;">
            <span class="ner-entity ner-brand">🏭 BRAND</span>
            <span class="ner-entity ner-product">👕 PRODUCT</span>
            <span class="ner-entity ner-location">📍 LOCATION</span>
            <span class="ner-entity ner-aspect">🏷️ ASPECT</span>
        </div>
        """, unsafe_allow_html=True)

        ner_input = st.text_area(
            "Masukkan teks review:",
            value="Saya beli kemeja di toko Zano cabang Jakarta, bahannya bagus dan jahitan rapi. Pramuniaganya juga ramah walau harganya mahal.",
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
                    row = {
                        'Teks': ent['text'],
                        'Label': ent['label'],
                        'Posisi Awal': ent['start'],
                        'Posisi Akhir': ent['end'],
                    }
                    if ent.get('confidence') is not None:
                        row['Confidence'] = f"{ent['confidence']*100:.1f}%"
                    ent_data.append(row)
                ent_df = pd.DataFrame(ent_data)
                st.dataframe(ent_df, use_container_width=True, hide_index=True)

        # Sample reviews NER
        st.markdown("---")
        st.markdown("#### 📝 Contoh Review dari Dataset:")

        try:
            train_df, _, _ = load_data()
            sample_reviews = train_df['text'].sample(5, random_state=42).tolist()
        except Exception:
            sample_reviews = [
                "Baju nya bagus banget, jahitan rapi dan bahan kain lembut",
                "Kemeja Erigo nyaman dipakai walau harga nya agak mahal",
                "Pelayanan kasir di mall Jakarta ramah dan diskon nya gede",
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
            price_counter = Counter()
            service_counter = Counter()
            from collections import defaultdict
            entity_type_counts = defaultdict(int)

            for text in all_data['text'].astype(str):
                ents = extract_entities(text)
                for ent in ents:
                    entity_type_counts[ent['label']] += 1
                    if ent['label'] == 'BRAND':
                        brand_counter[ent['text'].lower()] += 1
                    elif ent['label'] == 'PRODUCT':
                        product_counter[ent['text'].lower()] += 1
                    elif ent['label'] == 'LOCATION':
                        location_counter[ent['text'].lower()] += 1
                    elif ent['label'] == 'PRICE':
                        price_counter[ent['text'].lower()] += 1
                    elif ent['label'] in ('SERVICE', 'PROMOTION'):
                        service_counter[ent['text'].lower()] += 1
                    else:
                        aspect_counter[ent['text'].lower()] += 1

        import plotly.express as px

        # Entity type distribution
        type_df = pd.DataFrame([
            {'Tipe Entitas': k, 'Jumlah': v}
            for k, v in entity_type_counts.items()
            if v > 0
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
            st.markdown("#### 👕 Top Product")
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
        <p>Coba prediksi sentimen aspek & NER pada ulasan toko baju/pakaian</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Quick Train if needed ───
    @st.cache_resource
    def load_best_multilabel_model():
        """Load the best multilabel model from joblib (no retraining)."""
        try:
            import joblib
            base_path = os.path.join(os.path.dirname(__file__), 'models')
            data = joblib.load(os.path.join(base_path, 'Kelp2_best_multilabel_model.joblib'))
            pipeline = data.get('pipeline') or data.get('model') or data
            # Prioritize 'labels' key — contains all 13 labels including OUT_OF_TOPIC
            target_cols = data.get('labels') or data.get('target_cols') or []
            # Model is ClassifierChain + LogReg → uses predict_proba
            score_type = data.get('score_type', 'predict_proba') if isinstance(data, dict) else 'predict_proba'
            raw_thresholds = data.get('thresholds', [])
            # thresholds is a list aligned to labels order
            if isinstance(raw_thresholds, (list, np.ndarray)) and target_cols:
                thresholds = {col: float(t) for col, t in zip(target_cols, raw_thresholds)}
            elif isinstance(raw_thresholds, dict):
                thresholds = raw_thresholds
            else:
                thresholds = {}
            return pipeline, thresholds, target_cols, score_type
        except Exception as e:
            st.error(f"Gagal load model: {e}")
            return None, {}, [], 'predict_proba'


    st.markdown('<div class="section-header">✍️ Masukkan Review</div>', unsafe_allow_html=True)

    demo_input = st.text_area(
        "Tulis review produk/toko di sini:",
        value="",
        height=120,
        placeholder="Contoh: Baju nya bagus banget, jahitan rapi, tapi harganya lumayan mahal. Pramuniaganya di store cabang Jakarta ramah.",
        key="demo_input"
    )

    # Quick examples
    st.markdown("**💡 Atau pilih contoh:**")
    example_cols = st.columns(3)
    examples = [
        "Kaos uniqlo nyaman dipakai, kainnya lembut dan harga diskon lumayan murah",
        "Pelayanan di toko Zano jelek banget, kasirnya judes dan antrian panjang",
        "Koleksi dress nya lengkap dengan harga terjangkau, tapi fitting room nya kotor"
    ]
    def set_demo_input(text):
        st.session_state['demo_input'] = text

    for col, example in zip(example_cols, examples):
        with col:
            st.button(f"📝 {example[:40]}...", key=f"ex_{example[:10]}", on_click=set_demo_input, args=(example,))

    if demo_input or st.session_state.get('demo_input'):
        text_to_analyze = demo_input or st.session_state.get('demo_input', '')

        if text_to_analyze.strip():
            st.markdown("---")

            with st.spinner("🔄 Memproses prediksi..."):
                pipeline, thresholds, target_cols, score_type = load_best_multilabel_model()

                if pipeline is None:
                    st.error("❌ Model terbaik tidak ditemukan. Pastikan file `models/Kelp2_best_multilabel_model.joblib` tersedia.")
                    st.stop()

                # Predict using best pipeline
                import numpy as np
                try:
                    use_decision = (score_type == 'decision_function') or (not hasattr(pipeline, 'predict_proba'))
                    if use_decision:
                        scores = pipeline.decision_function([text_to_analyze])
                        scores = np.array(scores)
                        if scores.ndim == 1:
                            scores = scores.reshape(1, -1)
                        default_thresh = 0.0
                    else:
                        raw_proba = pipeline.predict_proba([text_to_analyze])
                        # ClassifierChain returns list of arrays (one per label)
                        if isinstance(raw_proba, list):
                            scores = np.array([[p[0][1] if p.shape[1] == 2 else p[0][0] for p in raw_proba]])
                        else:
                            if hasattr(raw_proba, 'toarray'):
                                raw_proba = raw_proba.toarray()
                            scores = np.array(raw_proba)
                        default_thresh = 0.5

                    if thresholds and len(thresholds) == scores.shape[1]:
                        thresh_arr = np.array([thresholds.get(col, default_thresh) for col in target_cols])
                        y_pred = (scores[0] >= thresh_arr).astype(int)
                    else:
                        y_pred = (scores[0] >= default_thresh).astype(int)
                except Exception as ex:
                    raw = pipeline.predict([text_to_analyze])
                    if hasattr(raw, 'toarray'):
                        raw = raw.toarray()
                    y_pred = np.array(raw).flatten()

                predictions = {}
                for idx, col in enumerate(target_cols):
                    if y_pred[idx] == 1:
                        # Skip non-aspect columns like OUT_OF_TOPIC
                        matched = False
                        for asp in ASPECT_COLUMNS:
                            if col.startswith(asp + '_'):
                                sentiment = col[len(asp) + 1:]
                                if asp not in predictions:
                                    predictions[asp] = []
                                predictions[asp].append(sentiment.lower())
                                matched = True
                                break
                        if not matched:
                            continue

                # NER
                entities = extract_entities(text_to_analyze)

            # ─── Map NER entities to aspects ───
            ASPECT_ENTITY_MAP = {
                'PRODUCT': {'PRODUCT', 'BRAND'},
                'PRICE': set(),
                'PLACE': {'LOCATION'},
                'PROMOTION': set(),
            }
            ASPECT_KEYWORD_MAP = {
                'PRODUCT': {'kualitas', 'bagus', 'jelek', 'awet', 'luntur', 'sobek', 'jahitan', 'koleksi', 'lengkap', 'ukuran', 'size', 'bahan', 'kain'},
                'PRICE': {'harga', 'murah', 'mahal'},
                'PLACE': {'tempat', 'bersih', 'kotor', 'luas', 'sempit', 'nyaman', 'parkir', 'fitting room', 'kamar ganti', 'toko', 'butik', 'outlet', 'store', 'cabang', 'mall', 'plaza'},
                'PROMOTION': {'diskon', 'promo', 'promosi', 'sale'},
            }

            def get_entities_for_aspect(aspect, entities):
                """Get NER entities relevant to a specific aspect."""
                relevant = []
                ner_labels = ASPECT_ENTITY_MAP.get(aspect, set())
                keywords = ASPECT_KEYWORD_MAP.get(aspect, set())
                for ent in entities:
                    if ent['label'] in ner_labels:
                        relevant.append(ent)
                    elif ent['label'] == 'ASPECT' and ent['text'].lower() in keywords:
                        relevant.append(ent)
                return relevant

            # ─── Display Results ───
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown('<div class="section-header">🏷️ Prediksi Sentimen Aspek</div>', unsafe_allow_html=True)

                for aspect in ASPECT_COLUMNS:
                    sentiments = predictions.get(aspect, [])
                    icon = ASPECT_DESCRIPTIONS[aspect]
                    aspect_entities = get_entities_for_aspect(aspect, entities)

                    if len(sentiments) == 0:
                        sent_badge = '<span style="padding: 4px 10px; border-radius: 6px; background: #f3f4f6; color: #6b7280; font-size: 0.8rem; font-weight: 600;">➖ Neutral/None</span>'
                    else:
                        sent_badge = ""
                        for sent in sentiments:
                            if sent == 'positive':
                                sent_badge += '<span style="margin-left: 4px; padding: 4px 10px; border-radius: 6px; background: #d1fae5; color: #065f46; font-size: 0.8rem; font-weight: 600;">✅ Positive</span>'
                            elif sent == 'negative':
                                sent_badge += '<span style="margin-left: 4px; padding: 4px 10px; border-radius: 6px; background: #fee2e2; color: #991b1b; font-size: 0.8rem; font-weight: 600;">❌ Negative</span>'
                            elif sent == 'neutral':
                                sent_badge += '<span style="margin-left: 4px; padding: 4px 10px; border-radius: 6px; background: #f3f4f6; color: #6b7280; font-size: 0.8rem; font-weight: 600;">➖ Neutral</span>'

                    # Build entity tags HTML
                    entity_tags_html = ""
                    if aspect_entities:
                        label_colors = {
                            'BRAND': ('#dbeafe', '#1e40af', '🏭'),
                            'PRODUCT': ('#dcfce7', '#166534', '👕'),
                            'LOCATION': ('#fef3c7', '#92400e', '📍'),
                            'ASPECT': ('#f3e8ff', '#6b21a8', '🏷️'),
                        }
                        tags = []
                        seen = set()
                        for ent in aspect_entities:
                            key = ent['text'].lower()
                            if key not in seen:
                                seen.add(key)
                                bg, tc, emoji = label_colors.get(ent['label'], ('#f3f4f6', '#333', '📌'))
                                tags.append(
                                    f'<span style="display:inline-block; background:{bg}; color:{tc}; '
                                    f'padding:2px 8px; border-radius:10px; font-size:0.72rem; font-weight:600; '
                                    f'margin:2px 3px 0 0;">{emoji} {ent["text"]}'
                                    f'<sup style="font-size:0.55rem; opacity:0.7; margin-left:2px;">{ent["label"]}</sup></span>'
                                )
                        entity_tags_html = f'<div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:2px;">{" ".join(tags)}</div>'

                    st.markdown(f"""
                    <div class="prediction-card" style="padding:0.8rem 1.2rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:600; font-size:0.95rem;">{icon}</span>
                            <div>{sent_badge}</div>
                        </div>
                        {entity_tags_html}
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
                            'PRODUCT': ('👕', '#dcfce7', '#166534'),
                            'LOCATION': ('📍', '#fef3c7', '#92400e'),
                            'ASPECT': ('🏷️', '#f3e8ff', '#6b21a8'),
                        }
                        emoji, bg, tc = label_colors.get(ent['label'], ('', '#f3f4f6', '#333'))
                        conf = ent.get('confidence')
                        conf_badge = (
                            f'<span style="margin-left:auto; font-size:0.72rem; font-weight:600; '
                            f'color:{tc}; opacity:0.85;">{conf*100:.0f}%</span>'
                            if conf is not None else ''
                        )
                        st.markdown(f"""
                        <div style="display:flex; align-items:center; gap:8px; margin:4px 0;
                                    padding:6px 10px; background:{bg}; border-radius:8px;">
                            <span>{emoji}</span>
                            <span style="font-weight:600; color:{tc};">{ent['text']}</span>
                            <span style="font-size:0.7rem; opacity:0.7;">({ent['label']})</span>
                            {conf_badge}
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
            text_labels = []
            colors = []
            for aspect in ASPECT_COLUMNS:
                sents = predictions.get(aspect, [])
                if 'positive' in sents and 'negative' in sents:
                    sentiment_scores.append(0.5)
                    text_labels.append("Mixed")
                    colors.append('#f59e0b')
                elif 'positive' in sents:
                    sentiment_scores.append(1)
                    text_labels.append("Positive")
                    colors.append('#00b09b')
                elif 'negative' in sents:
                    sentiment_scores.append(-1)
                    text_labels.append("Negative")
                    colors.append('#eb3349')
                else:
                    sentiment_scores.append(0.1) # so it shows up in radar slightly
                    text_labels.append("Neutral")
                    colors.append('#95a5a6')

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[abs(s) for s in sentiment_scores],
                theta=[ASPECT_DESCRIPTIONS[a] for a in ASPECT_COLUMNS],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='#667eea', width=2),
                marker=dict(
                    color=colors,
                    size=10,
                ),
                text=text_labels,
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
    <p>🔍 ABSA Dashboard — Multilabel Classification & Named Entity Recognition</p>
</div>
""", unsafe_allow_html=True)
