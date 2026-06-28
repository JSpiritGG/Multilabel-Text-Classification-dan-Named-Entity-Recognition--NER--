"""
==========================================================================
ABSA dari Review Google Places: Multilabel Text Classification & NER
==========================================================================
Streamlit Application - Mata Kuliah: Pemrosesan Bahasa Alami - UAS
==========================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ABSA - Multilabel Classification & NER",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── CSS ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
section[data-testid="stSidebar"] * { color: #e0e0ff !important; }

.hero-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 2rem;
    color: white; text-align: center;
    box-shadow: 0 20px 60px rgba(102,126,234,0.3);
}
.hero-container h1 { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }
.hero-container p  { font-size: 1.1rem; opacity: 0.9; font-weight: 300; }

.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 14px; padding: 1.5rem; text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.8);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 8px 25px rgba(0,0,0,0.12); }
.metric-card .metric-value {
    font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .metric-label { font-size: 0.85rem; color: #666; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }

.section-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 700; font-size: 1.5rem; margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem; border-bottom: 3px solid;
    border-image: linear-gradient(90deg, #667eea 0%, #764ba2 100%) 1;
}
.info-box {
    background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
    border-radius: 12px; padding: 1.2rem 1.5rem;
    border-left: 5px solid #667eea; margin: 1rem 0;
    font-size: 0.9rem; color: #312e81;
}
.prediction-card {
    background: white; border-radius: 14px; padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb; margin: 0.5rem 0;
    transition: all 0.3s ease;
}
.prediction-card:hover { box-shadow: 0 8px 30px rgba(0,0,0,0.12); }

.ner-entity { display: inline; padding: 0.2rem 0.5rem; border-radius: 6px; font-weight: 600; margin: 0 2px; }
.ner-brand    { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
.ner-product  { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.ner-location { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.ner-aspect   { background: #f3e8ff; color: #6b21a8; border: 1px solid #c084fc; }

.footer { text-align: center; padding: 2rem 0 1rem; color: #9ca3af; font-size: 0.8rem; border-top: 1px solid #e5e7eb; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────── Constants ───────────────────
ASPECT_COLUMNS    = ['PRODUCT', 'PRICE', 'PLACE', 'PROMOTION']
SENTIMENT_LABELS  = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
ASPECT_DESCRIPTIONS = {
    'PRODUCT':   '👕 Produk (Product)',
    'PRICE':     '💰 Harga (Price)',
    'PLACE':     '🏪 Tempat/Toko (Place)',
    'PROMOTION': '🏷️ Promosi (Promotion)',
}

# ─────────────────── Data Loading ───────────────────
@st.cache_data
def load_data():
    base_path = os.path.join(os.path.dirname(__file__), 'dataset')
    train_df = pd.read_csv(os.path.join(base_path, 'Kelp2_multilabel_train.csv'))
    valid_df = pd.read_csv(os.path.join(base_path, 'Kelp2_multilabel_val.csv'))
    test_df  = pd.read_csv(os.path.join(base_path, 'Kelp2_multilabel_test.csv'))
    return train_df, valid_df, test_df

# ─────────────────── Best Model Loaders ───────────────────
@st.cache_resource
def load_best_multilabel_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'Kelp2_best_multilabel_model.joblib')
        data = joblib.load(model_path)
        pipeline    = data.get('pipeline') or data.get('model')
        target_cols = data.get('target_cols') or [f"{a}{s}" for a in ASPECT_COLUMNS for s in SENTIMENT_LABELS]
        raw_thresh  = data.get('thresholds', {})
        if isinstance(raw_thresh, (list, np.ndarray)):
            thresholds = {col: float(t) for col, t in zip(target_cols, raw_thresh)}
        elif isinstance(raw_thresh, dict):
            thresholds = raw_thresh
        else:
            thresholds = {}
        # detect score type
        try:
            _ = pipeline.decision_function(["test"])
            score_type = 'decision'
        except Exception:
            score_type = 'proba'
        return pipeline, thresholds, target_cols, score_type
    except Exception as e:
        return None, {}, [], 'decision'

@st.cache_resource
def load_ner_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'Kelp2_best_ner_model.joblib')
        data = joblib.load(model_path)
        clf        = data.get('classifier')
        vectorizer = data.get('vectorizer') or data.get('dict_vectorizer')
        all_tags   = data.get('all_tags') or data.get('labels')
        return clf, vectorizer, all_tags
    except Exception:
        return None, None, None

# ─────────────────── NER Helpers ───────────────────
BRAND_ENTITIES = {
    'zano','hos of shopaholic','hos','shopaholic','uniqlo','zara','h&m',
    'matahari','ramayana','borma','erigo','3second','greenlight',
    'levis','adidas','nike','puma','champion'
}
PRODUCT_ENTITIES = {
    'baju','kaos','kemeja','celana','rok','jaket','sweater','hoodie',
    'dress','gamis','tunik','blouse','jas','pakaian','outfit',
    'jeans','chino','kulot','legging','kain','bahan'
}
LOCATION_ENTITIES = {
    'jakarta','bandung','surabaya','bali','jogja','yogyakarta','semarang',
    'medan','makassar','palembang','malang','solo','denpasar','tasikmalaya',
    'mall','plaza','toko','butik','outlet','store','cabang'
}
ASPECT_TERM_ENTITIES = {
    'harga','murah','mahal','diskon','promo','promosi','sale',
    'kualitas','bagus','jelek','awet','luntur','sobek','jahitan',
    'pelayanan','ramah','kasir','pramuniaga','mbak','mas','layan',
    'tempat','bersih','kotor','luas','sempit','nyaman','parkir',
    'fitting room','kamar ganti','koleksi','lengkap','ukuran','size'
}

def _extract_token_features(tokens, idx):
    t = tokens[idx]
    tl = t.lower()
    feats = {
        'word': tl,
        'is_upper': t.isupper(),
        'is_title': t.istitle(),
        'prefix2': tl[:2],
        'prefix3': tl[:3],
        'suffix2': tl[-2:],
        'suffix3': tl[-3:],
        'prev1': tokens[idx-1].lower() if idx > 0 else '<START>',
        'next1': tokens[idx+1].lower() if idx < len(tokens)-1 else '<END>',
        'prev2': tokens[idx-2].lower() if idx > 1 else '<START2>',
        'next2': tokens[idx+2].lower() if idx < len(tokens)-2 else '<END2>',
        'bigram_l': (tokens[idx-1].lower() + '_' + tl) if idx > 0 else '<S>_' + tl,
        'bigram_r': (tl + '_' + tokens[idx+1].lower()) if idx < len(tokens)-1 else tl + '_<E>',
    }
    return feats

def extract_entities_ml(text, clf, vectorizer, all_tags):
    tokens = text.split()
    if not tokens:
        return []
    feats_list = [_extract_token_features(tokens, i) for i in range(len(tokens))]
    try:
        X = vectorizer.transform(feats_list)
        preds = clf.predict(X)
    except Exception:
        return extract_entities_fallback(text)

    entities = []
    current_entity = None
    current_start  = 0
    char_pos = 0
    for i, (token, tag) in enumerate(zip(tokens, preds)):
        token_start = text.find(token, char_pos)
        token_end   = token_start + len(token)
        char_pos    = token_end

        label = _bio_to_display(tag)
        if tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {'start': token_start, 'end': token_end, 'label': label, 'text': token}
            current_start  = token_start
        elif tag.startswith('I-') and current_entity and current_entity['label'] == label:
            current_entity['end']  = token_end
            current_entity['text'] = text[current_start:token_end]
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    return entities

def _bio_to_display(tag):
    tag = tag.replace('B-','').replace('I-','')
    if 'PRODUCT' in tag:  return 'PRODUCT'
    if 'PLACE'   in tag or 'LOCATION' in tag: return 'LOCATION'
    if 'PRICE'   in tag:  return 'PRICE'
    if 'SERVICE' in tag or 'PROMOTION' in tag: return 'ASPECT'
    return 'ASPECT'

def extract_entities_fallback(text):
    text_lower = text.lower()
    entities   = []
    def add_entity(entity_type, word):
        for match in re.finditer(rf'\b{re.escape(word)}\b', text_lower):
            start, end = match.start(), match.end()
            if not any(start < e['end'] and end > e['start'] for e in entities):
                entities.append({'start': start, 'end': end, 'label': entity_type, 'text': text[start:end]})
    for w in sorted(BRAND_ENTITIES, key=len, reverse=True):    add_entity('BRAND', w)
    for w in sorted(PRODUCT_ENTITIES, key=len, reverse=True):  add_entity('PRODUCT', w)
    for w in sorted(LOCATION_ENTITIES, key=len, reverse=True): add_entity('LOCATION', w)
    for w in sorted(ASPECT_TERM_ENTITIES, key=len, reverse=True): add_entity('ASPECT', w)
    return sorted(entities, key=lambda x: x['start'])

def extract_entities(text):
    clf, vectorizer, all_tags = load_ner_model()
    if clf is not None and vectorizer is not None:
        return extract_entities_ml(text, clf, vectorizer, all_tags)
    return extract_entities_fallback(text)

def render_ner_html(text, entities):
    if not entities:
        return f'<p style="font-size:1rem;line-height:2.2">{text}</p>'
    color_map = {
        'BRAND':    ('#dbeafe','#93c5fd','#1e40af'),
        'PRODUCT':  ('#dcfce7','#86efac','#166534'),
        'LOCATION': ('#fef3c7','#fcd34d','#92400e'),
        'ASPECT':   ('#f3e8ff','#c084fc','#6b21a8'),
        'PRICE':    ('#fef3c7','#fcd34d','#92400e'),
    }
    parts, last = [], 0
    for ent in entities:
        bg, bd, tc = color_map.get(ent['label'], ('#f3f4f6','#e5e7eb','#374151'))
        if ent['start'] > last:
            parts.append(text[last:ent['start']])
        parts.append(
            f'<span style="background:{bg};color:{tc};border:1px solid {bd};'
            f'padding:2px 6px;border-radius:4px;font-weight:600;margin:0 1px">'
            f'{text[ent["start"]:ent["end"]]}'
            f'<sup style="font-size:0.6rem;margin-left:2px;opacity:0.8">{ent["label"]}</sup></span>'
        )
        last = ent['end']
    if last < len(text):
        parts.append(text[last:])
    return f'<p style="font-size:1rem;line-height:2.2">{"".join(parts)}</p>'

# ─────────────────── Sidebar ───────────────────
with st.sidebar:
    st.markdown("## 🔍 ABSA Navigator")
    st.markdown("---")
    page = st.radio(
        "Pilih Halaman",
        ["🏷️ Multilabel Classification", "📛 Named Entity Recognition", "🚀 Demo Prediksi"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
<div style="text-align:center;opacity:0.7;font-size:0.75rem">
<p>ABSA Dashboard</p>
<p>Multilabel Classification & NER</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  PAGE 1 — MULTILABEL CLASSIFICATION
# ═══════════════════════════════════════════════════
if page == "🏷️ Multilabel Classification":
    st.markdown("""
<div class="hero-container" style="padding:1.5rem">
<h1 style="font-size:1.8rem">🏷️ Multilabel Text Classification</h1>
<p>Training dan evaluasi model klasifikasi multilabel ABSA</p>
</div>""", unsafe_allow_html=True)

    try:
        train_df, valid_df, test_df = load_data()
    except FileNotFoundError:
        st.error("Dataset tidak ditemukan. Pastikan file CSV berada di folder `dataset/`.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["⚙️ Konfigurasi Training", "📊 Hasil Evaluasi", "🔎 Analisis Detail"])

    with tab1:
        st.markdown('<div class="section-header">Konfigurasi Model</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Feature Extraction**")
            feature_method = st.selectbox("Metode Ekstraksi Fitur:", ["TF-IDF", "Count Vectorizer", "TF-IDF N-gram"])
            if feature_method == "TF-IDF N-gram":
                max_features = st.slider("Max Features:", 500, 10000, 5000, 500)
                ngram_range  = st.selectbox("N-gram Range:", ["(1,1)", "(1,2)", "(1,3)"])
            else:
                max_features = st.slider("Max Features:", 500, 10000, 3000, 500)
                ngram_range  = "(1,1)"
        with col2:
            st.markdown("**Model Selection**")
            model_type         = st.selectbox("Algoritma Klasifikasi:", ["Support Vector Machine (SVM)", "Random Forest", "Logistic Regression", "Naive Bayes"])
            multilabel_strategy = st.selectbox("Strategi Multilabel:", ["Binary Relevance", "Classifier Chain", "Label Powerset"])
        st.markdown("---")

        if st.button("🚀 Mulai Training", type="primary", use_container_width=True):
            with st.spinner("Sedang melatih model..."):
                import time
                from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
                from sklearn.svm import LinearSVC
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.naive_bayes import MultinomialNB
                from sklearn.multiclass import OneVsRestClassifier
                from sklearn.multioutput import ClassifierChain as SklearnChain
                from sklearn.metrics import (classification_report, f1_score,
                                             precision_score, recall_score, hamming_loss,
                                             accuracy_score)

                progress_bar = st.progress(0, text="Mempersiapkan data...")

                ngram = eval(ngram_range)
                if feature_method == "TF-IDF":
                    vectorizer = TfidfVectorizer(max_features=max_features)
                elif feature_method == "TF-IDF N-gram":
                    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram)
                else:
                    vectorizer = CountVectorizer(max_features=max_features)

                X_train = vectorizer.fit_transform(train_df['text'].astype(str))
                X_test  = vectorizer.transform(test_df['text'].astype(str))
                progress_bar.progress(20, text="Fitur berhasil diekstrak...")

                target_cols = [f"{a}{s}" for a in ASPECT_COLUMNS for s in SENTIMENT_LABELS
                               if f"{a}{s}" in train_df.columns]
                y_train = train_df[target_cols].values
                y_test  = test_df[target_cols].values
                progress_bar.progress(30, text=f"Melatih model dengan {multilabel_strategy}...")

                # Base classifier
                if model_type == "Support Vector Machine (SVM)":
                    base_clf = LinearSVC(max_iter=10000, random_state=42, class_weight='balanced')
                elif model_type == "Random Forest":
                    base_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
                elif model_type == "Logistic Regression":
                    base_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
                else:
                    base_clf = MultinomialNB()

                # Multilabel strategy — sklearn native only (no skmultilearn needed)
                if multilabel_strategy == "Binary Relevance":
                    clf = OneVsRestClassifier(base_clf)
                elif multilabel_strategy == "Classifier Chain":
                    clf = SklearnChain(base_clf, order='random', random_state=42)
                else:
                    # Label Powerset fallback
                    clf = OneVsRestClassifier(base_clf)
                    st.info("ℹ️ Label Powerset tidak tersedia di scikit-learn standar. Menggunakan Binary Relevance sebagai pengganti.")

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                if hasattr(y_pred, 'toarray'):
                    y_pred_array = y_pred.toarray()
                else:
                    y_pred_array = np.array(y_pred)
                progress_bar.progress(90, text="Menghitung metrik evaluasi...")

                report = classification_report(y_test, y_pred_array, target_names=target_cols,
                                               output_dict=True, zero_division=0)
                results = {
                    'accuracy':     accuracy_score(y_test, y_pred_array),
                    'f1_macro':     f1_score(y_test, y_pred_array, average='macro', zero_division=0),
                    'f1_weighted':  f1_score(y_test, y_pred_array, average='weighted', zero_division=0),
                    'precision':    precision_score(y_test, y_pred_array, average='weighted', zero_division=0),
                    'recall':       recall_score(y_test, y_pred_array, average='weighted', zero_division=0),
                    'hamming_loss': hamming_loss(y_test, y_pred_array),
                    'report':       report,
                    'y_true':       y_test.tolist(),
                    'y_pred':       y_pred_array.tolist(),
                    'target_cols':  target_cols,
                }
                time.sleep(0.3)
                progress_bar.progress(100, text="Training selesai!")
                time.sleep(0.3)
                progress_bar.empty()

                st.session_state['classification_results'] = results
                st.session_state['vectorizer']   = vectorizer
                st.session_state['model_type']   = model_type
                st.session_state['feature_method'] = feature_method
                st.success("✅ Model berhasil dilatih! Lihat hasil di tab **Hasil Evaluasi**.")

    with tab2:
        st.markdown('<div class="section-header">Hasil Evaluasi Model</div>', unsafe_allow_html=True)
        if 'classification_results' not in st.session_state:
            st.info("Belum ada model yang dilatih. Silakan konfigurasi dan latih model di tab **Konfigurasi Training**.")
        else:
            results     = st.session_state['classification_results']
            target_cols = results['target_cols']
            report      = results['report']

            st.markdown(f"Model: `{st.session_state.get('model_type','NA')}` | Fitur: `{st.session_state.get('feature_method','NA')}`")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Exact Match Accuracy", f"{results['accuracy']:.4f}")
            c2.metric("F1 Macro",             f"{results['f1_macro']:.4f}")
            c3.metric("F1 Weighted",          f"{results['f1_weighted']:.4f}")
            c4.metric("Hamming Loss",         f"{results['hamming_loss']:.4f}")

            f1s  = [report[col]['f1-score']  for col in target_cols]
            prec = [report[col]['precision'] for col in target_cols]
            recs = [report[col]['recall']    for col in target_cols]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='F1-Score',  x=target_cols, y=f1s,  marker_color='#667eea'))
            fig.add_trace(go.Bar(name='Precision', x=target_cols, y=prec, marker_color='#764ba2'))
            fig.add_trace(go.Bar(name='Recall',    x=target_cols, y=recs, marker_color='#f093fb'))
            fig.update_layout(barmode='group', title='Metrik per Target Aspek-Sentimen',
                              yaxis=dict(range=[0,1], title='Skor'),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(family='Inter'), height=500, xaxis_tickangle=-45,
                              legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
            st.plotly_chart(fig, use_container_width=True)

            summary = [{'Target': col, 'Precision': f"{report[col]['precision']:.4f}",
                        'Recall': f"{report[col]['recall']:.4f}",
                        'F1-Score': f"{report[col]['f1-score']:.4f}",
                        'Support': int(report[col]['support'])} for col in target_cols]
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown('<div class="section-header">Analisis Detail per Target</div>', unsafe_allow_html=True)
        if 'classification_results' not in st.session_state:
            st.info("Belum ada model yang dilatih.")
        else:
            from sklearn.metrics import confusion_matrix
            results     = st.session_state['classification_results']
            target_cols = results['target_cols']
            selected    = st.selectbox("Pilih target aspek-sentimen:", target_cols, key='detail_target')
            idx         = target_cols.index(selected)
            y_true_col  = [row[idx] for row in results['y_true']]
            y_pred_col  = [row[idx] for row in results['y_pred']]
            cm = confusion_matrix(y_true_col, y_pred_col, labels=[0,1])
            fig = px.imshow(cm, x=['Predicted 0','Predicted 1'], y=['Actual 0','Actual 1'],
                            color_continuous_scale='Blues', text_auto=True,
                            title=f'Confusion Matrix — {selected}')
            fig.update_layout(xaxis_title='Predicted', yaxis_title='Actual',
                              font=dict(family='Inter'), height=400)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════
#  PAGE 2 — NAMED ENTITY RECOGNITION
# ═══════════════════════════════════════════════════
elif page == "📛 Named Entity Recognition":
    st.markdown("""
<div class="hero-container" style="padding:1.5rem">
<h1 style="font-size:1.8rem">📛 Named Entity Recognition</h1>
<p>Identifikasi entitas bernama dalam review toko/pakaian</p>
</div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔍 NER Interaktif", "📊 Statistik Entitas"])

    with tab1:
        st.markdown('<div class="section-header">Ekstraksi Entitas</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1rem">
<span class="ner-entity ner-brand">BRAND</span>
<span class="ner-entity ner-product">PRODUCT</span>
<span class="ner-entity ner-location">LOCATION</span>
<span class="ner-entity ner-aspect">ASPECT</span>
</div>""", unsafe_allow_html=True)

        ner_input = st.text_area("Masukkan teks review:",
            value="Saya beli kemeja di toko Zano cabang Jakarta, bahannya bagus dan jahitan rapi. Pramuniaganya juga ramah walau harganya mahal.",
            height=120, key='ner_input')

        if st.button("🔍 Ekstrak Entitas", type="primary", key='ner_btn'):
            entities = extract_entities(ner_input)
            ner_html = render_ner_html(ner_input, entities)
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("**Hasil NER**")
            st.markdown(ner_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if entities:
                ent_df = pd.DataFrame([{'Teks': e['text'], 'Label': e['label'],
                                        'Start': e['start'], 'End': e['end']} for e in entities])
                st.markdown("**Entitas Terdeteksi**")
                st.dataframe(ent_df, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada entitas terdeteksi.")

    with tab2:
        st.markdown('<div class="section-header">Statistik Entitas dalam Dataset</div>', unsafe_allow_html=True)
        try:
            train_df, valid_df, test_df = load_data()
        except Exception:
            st.error("Dataset tidak ditemukan.")
            st.stop()


        with st.spinner("Menganalisis entitas dalam seluruh dataset..."):
            all_data    = pd.concat([train_df, valid_df, test_df], ignore_index=True)
            brand_c     = Counter()
            product_c   = Counter()
            location_c  = Counter()
            aspect_c    = Counter()
            type_counts  = {'BRAND': 0, 'PRODUCT': 0, 'LOCATION': 0, 'ASPECT': 0}

            for text in all_data['text'].astype(str):
                ents = extract_entities_fallback(text)
                for e in ents:
                    type_counts[e['label']] = type_counts.get(e['label'], 0) + 1
                    if e['label'] == 'BRAND':    brand_c[e['text'].lower()]    += 1
                    elif e['label'] == 'PRODUCT': product_c[e['text'].lower()] += 1
                    elif e['label'] == 'LOCATION': location_c[e['text'].lower()] += 1
                    elif e['label'] == 'ASPECT':  aspect_c[e['text'].lower()]  += 1

        type_df = pd.DataFrame([{'Tipe': k, 'Jumlah': v} for k,v in type_counts.items()])
        fig_pie = px.pie(type_df, values='Jumlah', names='Tipe',
                         color_discrete_sequence=['#667eea','#43e97b','#fcd34d','#c084fc'],
                         title='Distribusi Tipe Entitas', hole=0.4)
        fig_pie.update_layout(font=dict(family='Inter'), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

        col1, col2 = st.columns(2)
        def bar_chart(counter, title, color):
            df = pd.DataFrame(counter.most_common(10), columns=['Item','Count'])
            fig = px.bar(df, x='Count', y='Item', orientation='h',
                         color_discrete_sequence=[color], title=title)
            fig.update_layout(yaxis=dict(autorange='reversed'), height=350,
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(family='Inter'))
            return fig

        with col1:
            if brand_c:    st.plotly_chart(bar_chart(brand_c,    'Top Brand',    '#667eea'), use_container_width=True)
            if location_c: st.plotly_chart(bar_chart(location_c, 'Top Location', '#fcd34d'), use_container_width=True)
        with col2:
            if product_c:  st.plotly_chart(bar_chart(product_c,  'Top Product',  '#43e97b'), use_container_width=True)
            if aspect_c:   st.plotly_chart(bar_chart(aspect_c,   'Top Aspect',   '#c084fc'), use_container_width=True)

# ═══════════════════════════════════════════════════
#  PAGE 3 — DEMO PREDIKSI
# ═══════════════════════════════════════════════════
elif page == "🚀 Demo Prediksi":
    st.markdown("""
<div class="hero-container" style="padding:1.5rem">
<h1 style="font-size:1.8rem">🚀 Demo Prediksi ABSA</h1>
<p>Prediksi sentimen aspek & ekstraksi entitas dari review toko/pakaian</p>
</div>""", unsafe_allow_html=True)

    pipeline, thresholds, target_cols, score_type = load_best_multilabel_model()
    if pipeline is None:
        st.error("❌ Model multilabel tidak ditemukan di `models/Kelp2_best_multilabel_model.joblib`. Pastikan file model tersedia.")
        st.stop()

    st.markdown('<div class="section-header">Masukkan Review</div>', unsafe_allow_html=True)
    demo_input = st.text_area("Tulis review produk/toko di sini:",
        value="", height=120, placeholder="Contoh: Baju nya bagus banget, jahitan rapi, tapi harganya lumayan mahal. Pramuniaganya ramah.",
        key='demo_input')

    ex_cols = st.columns(3)
    examples = [
        "Kaos uniqlo nyaman dipakai, kainnya lembut dan harga diskon lumayan murah",
        "Pelayanan di toko Zano jelek banget, kasirnya judes dan antrian panjang",
        "Koleksi dress nya lengkap dengan harga terjangkau, tapi fitting room nya kotor",
    ]
    def set_demo(text): st.session_state['demo_input'] = text
    for col, ex in zip(ex_cols, examples):
        with col:
            st.button(ex[:40]+"...", key=f"ex_{ex[:10]}", on_click=set_demo, args=(ex,))

    text_to_analyze = st.session_state.get('demo_input', '') or demo_input

    if text_to_analyze.strip():
        st.markdown("---")
        with st.spinner("Memproses prediksi..."):
            # Multilabel prediction using best model
            if score_type == 'decision':
                scores = pipeline.decision_function([text_to_analyze])
            else:
                scores = pipeline.predict_proba([text_to_analyze])
            scores = np.array(scores)
            if scores.ndim == 1:
                scores = scores.reshape(1, -1)

            predictions = {}
            for i, col in enumerate(target_cols):
                score   = float(scores[0][i]) if scores.shape[1] > i else 0.0
                thresh  = thresholds.get(col, 0.0 if score_type == 'decision' else 0.5)
                if score >= thresh:
                    parts   = col.split('_') if '_' in col else [col[:len(col)//2], col[len(col)//2:]]
                    aspect  = parts[0]
                    sentiment = parts[1].lower() if len(parts) > 1 else 'positive'
                    if aspect not in predictions:
                        predictions[aspect] = []
                    predictions[aspect].append(sentiment)

            entities = extract_entities(text_to_analyze)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<div class="section-header">Prediksi Sentimen Aspek</div>', unsafe_allow_html=True)
            sentiment_scores, text_labels, radar_colors = [], [], []
            for aspect in ASPECT_COLUMNS:
                sents = predictions.get(aspect, [])
                icon  = ASPECT_DESCRIPTIONS[aspect]
                if 'positive' in sents and 'negative' in sents:
                    badge = '<span style="margin-left:4px;padding:4px 10px;border-radius:6px;background:#fef3c7;color:#92400e;font-size:0.8rem;font-weight:600">Mixed</span>'
                    sentiment_scores.append(0.5); text_labels.append('Mixed'); radar_colors.append('#f59e0b')
                elif 'positive' in sents:
                    badge = '<span style="margin-left:4px;padding:4px 10px;border-radius:6px;background:#d1fae5;color:#065f46;font-size:0.8rem;font-weight:600">Positive</span>'
                    sentiment_scores.append(1); text_labels.append('Positive'); radar_colors.append('#00b09b')
                elif 'negative' in sents:
                    badge = '<span style="margin-left:4px;padding:4px 10px;border-radius:6px;background:#fee2e2;color:#991b1b;font-size:0.8rem;font-weight:600">Negative</span>'
                    sentiment_scores.append(-1); text_labels.append('Negative'); radar_colors.append('#eb3349')
                else:
                    badge = '<span style="padding:4px 10px;border-radius:6px;background:#f3f4f6;color:#6b7280;font-size:0.8rem;font-weight:600">Neutral/None</span>'
                    sentiment_scores.append(0.1); text_labels.append('Neutral'); radar_colors.append('#95a5a6')

                st.markdown(f"""
<div class="prediction-card" style="padding:0.8rem 1.2rem">
<div style="display:flex;justify-content:space-between;align-items:center">
<span style="font-weight:600;font-size:0.95rem">{icon}</span>
<div>{badge}</div>
</div>
</div>""", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-header">Entitas Terdeteksi (NER)</div>', unsafe_allow_html=True)
            ner_html = render_ner_html(text_to_analyze, entities)
            st.markdown(f'<div class="prediction-card">{ner_html}</div>', unsafe_allow_html=True)
            if entities:
                color_map_label = {'BRAND':'🔵','PRODUCT':'🟢','LOCATION':'🟡','ASPECT':'🟣','PRICE':'🟡'}
                for ent in entities:
                    emoji = color_map_label.get(ent['label'],'⚪')
                    st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin:4px 0;padding:6px 10px;background:#f9fafb;border-radius:8px">
<span>{emoji}</span>
<span style="font-weight:600">{ent["text"]}</span>
<span style="font-size:0.7rem;opacity:0.7">{ent["label"]}</span>
</div>""", unsafe_allow_html=True)
            else:
                st.info("Tidak ada entitas terdeteksi.")

        # Radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[abs(s) for s in sentiment_scores],
            theta=[ASPECT_DESCRIPTIONS[a] for a in ASPECT_COLUMNS],
            fill='toself', fillcolor='rgba(102,126,234,0.2)',
            line=dict(color='#667eea', width=2),
            marker=dict(color=radar_colors, size=10),
            text=text_labels, hovertemplate='%{theta}<br>Sentimen: %{text}<extra></extra>',
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1.2])),
            showlegend=False, title='Radar Chart Sentimen Aspek',
            font=dict(family='Inter'), height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────── Footer ───────────────────
st.markdown("""
<div class="footer">
<p>ABSA Dashboard — Multilabel Classification & Named Entity Recognition</p>
</div>""", unsafe_allow_html=True)
