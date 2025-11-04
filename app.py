# app.py — Interface Streamlit (BoW/TF-IDF + NB/SVM) com ablação e F1-macro + matriz de confusão
import re, unicodedata
import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer, SnowballStemmer, WordNetLemmatizer

# ---------- limpeza ----------
URL_RE     = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NUM_RE     = re.compile(r"\b\d+\b")

def basic_clean(text, lower=True, strip_accents=True):
    if not isinstance(text, str): text = str(text)
    t = text
    if lower:
        t = t.lower()
    if strip_accents:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = URL_RE.sub(" <url> ", t)
    t = MENTION_RE.sub(" <user> ", t)
    t = HASHTAG_RE.sub(lambda m: " " + m.group(0)[1:] + " ", t)  # mantém a palavra da hashtag
    t = re.sub(r"[^\w\s\.\,\!\?\-<>/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_tokenizer(use_stemming=False, use_lemmatization=False, language="english"):
    if language.lower().startswith("port"):
        stop = set(stopwords.words("portuguese")); stemmer = RSLPStemmer(); lemmatizer = None
    else:
        stop = set(stopwords.words("english")); stemmer = SnowballStemmer("english"); lemmatizer = WordNetLemmatizer()

    def tokenize(doc):
        toks = re.findall(r"\b\w+\b", doc)
        out = []
        for tok in toks:
            if tok in stop: continue
            if use_stemming: tok = stemmer.stem(tok)
            elif use_lemmatization and lemmatizer is not None: tok = lemmatizer.lemmatize(tok)
            out.append(tok)
        return out
    return tokenize

def build_vectorizer(kind="tfidf", tokenizer=None, ngram_max=1, max_features=30000):
    if kind == "bow":
        return CountVectorizer(tokenizer=tokenizer, ngram_range=(1, ngram_max), max_features=max_features)
    return TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, ngram_max), max_features=max_features, sublinear_tf=True)

def build_model(name="NB", class_weight=None):
    return MultinomialNB(alpha=1.0) if name=="NB" else LinearSVC(C=1.0, class_weight=class_weight, dual=True)

def evaluate(y_true, y_pred, labels_order=None):
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    rep = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return f1, cm, rep

def plot_confusion(cm, classes):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Matriz de Confusão")
    ax.set_xticks(np.arange(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_ylabel("Verdadeiro"); ax.set_xlabel("Predito")
    fig.tight_layout()
    return fig

def run_experiment(train_df, valid_df, vec_type="tfidf", ngram_max=1,
                   use_stemming=False, use_lemmatization=False, use_smote=False,
                   models=("NB","SVM"), language="english", class_weight_svm="balanced"):
    # As bases já têm [0,1,2,3] -> usamos 3=texto, 2=sentimento
    df_train = train_df[[3,2]].dropna().copy()
    df_valid = valid_df[[3,2]].dropna().copy()
    df_train[3] = df_train[3].apply(basic_clean)
    df_valid[3] = df_valid[3].apply(basic_clean)

    le = LabelEncoder()
    y_train = le.fit_transform(df_train[2].values); X_train = df_train[3].values
    y_test  = le.transform(df_valid[2].values);     X_test  = df_valid[3].values
    labels_order = list(range(len(le.classes_)))

    tokenizer = build_tokenizer(use_stemming, use_lemmatization, language)
    vectorizer = build_vectorizer(vec_type, tokenizer, ngram_max)

    results = []
    for m in models:
        clf = build_model("SVM", class_weight=class_weight_svm) if m=="SVM" else build_model("NB")
        pipe = ImbPipeline([("vec", vectorizer), ("smote", SMOTE()) , ("clf", clf)]) if use_smote \
               else Pipeline([("vec", vectorizer), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        f1, cm, rep = evaluate(y_test, y_pred, labels_order)
        results.append({"modelo":m,"vetorizacao":vec_type,"ngram_max":ngram_max,
                        "stemming":use_stemming,"lemmatization":use_lemmatization,"smote":use_smote,
                        "f1_macro":f1,"report":rep,"cm":cm,"classes":le.classes_,"pipeline":pipe})
    return results

# ---------------- UI ----------------
st.set_page_config(page_title="DM/ML - Sentimento (TF-IDF/BoW + NB/SVM)", layout="wide")
st.title("Projeto DM/ML — Classificação de Sentimentos (Baselines)")

st.caption("Treina em twitter_training.csv e avalia em twitter_validation.csv já incluídos no repositório.")

# Opções padrão para seu dataset (inglês)
vec_type   = st.selectbox("Representação", ["tfidf","bow"], index=0)
ngram_max  = st.selectbox("n-gramas (máx.)", [1,2], index=0)
use_stem   = st.checkbox("Stemming", value=False)
use_lemma  = st.checkbox("Lematização", value=False)
use_smote  = st.checkbox("Aplicar SMOTE", value=False)
models     = st.multiselect("Modelos", ["NB","SVM"], default=["NB","SVM"])
lang       = st.selectbox("Idioma (stopwords/stemmer)", ["english","portuguese"], index=0)
cw_bal     = st.checkbox('SVM com class_weight="balanced"', value=True)

if st.button("Rodar experimento"):
    try:
        train_df = pd.read_csv("twitter_training.csv", header=None)
        valid_df = pd.read_csv("twitter_validation.csv", header=None)
    except Exception as e:
        st.error(f"Erro lendo CSVs: {e}")
        st.stop()
    results = run_experiment(train_df, valid_df, vec_type, ngram_max,
                             use_stem, use_lemma, use_smote,
                             tuple(models) if models else ("NB",),
                             language=lang, class_weight_svm=("balanced" if cw_bal else None))
    st.subheader("Resumo")
    st.dataframe(pd.DataFrame([{
        "Modelo":r["modelo"], "Vetorização":r["vetorizacao"], "n-grama máx":r["ngram_max"],
        "Stemming":r["stemming"], "Lemma":r["lemmatization"], "SMOTE":r["smote"],
        "F1-macro":round(r["f1_macro"],4)
    } for r in results]), use_container_width=True)

    for r in results:
        st.markdown(f"### {r['modelo']} — detalhes")
        fig = plot_confusion(r["cm"], r["classes"])
        st.pyplot(fig)
        st.dataframe(pd.DataFrame(r["report"]).transpose(), use_container_width=True)
