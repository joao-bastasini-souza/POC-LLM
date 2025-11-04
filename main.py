# main.py — runner de console (sem Streamlit)
import re, unicodedata, pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

URL_RE=re.compile(r"https?://\S+|www\.\S+"); MENTION_RE=re.compile(r"@\w+"); HASHTAG_RE=re.compile(r"#\w+")
def clean(t:str)->str:
    if not isinstance(t,str): t=str(t)
    t=t.lower()
    t="".join(c for c in unicodedata.normalize("NFD",t) if unicodedata.category(c)!="Mn")
    t=URL_RE.sub(" <url> ",t); t=MENTION_RE.sub(" <user> ",t); t=HASHTAG_RE.sub(lambda m:" "+m.group(0)[1:]+" ",t)
    t=re.sub(r"[^\w\s\.\,\!\?\-<>/]", " ", t)
    t=re.sub(r"\s+"," ",t).strip()
    return t

def run():
    train=pd.read_csv("twitter_training.csv", header=None)[[3,2]].dropna()
    valid=pd.read_csv("twitter_validation.csv", header=None)[[3,2]].dropna()
    train[3]=train[3].apply(clean); valid[3]=valid[3].apply(clean)

    le=LabelEncoder()
    y_tr=le.fit_transform(train[2].values); X_tr=train[3].values
    y_te=le.transform(valid[2].values);    X_te=valid[3].values

    vec=TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True, max_features=30000)

    for name,clf in [("NB",MultinomialNB()),("SVM",LinearSVC(class_weight="balanced"))]:
        pipe=Pipeline([("vec",vec),("clf",clf)])
        pipe.fit(X_tr,y_tr); pred=pipe.predict(X_te)
        f1=f1_score(y_te,pred,average="macro")
        print(f"\n=== {name} (TF-IDF) ===")
        print(f"F1-macro: {f1:.4f}")
        print(confusion_matrix(y_te,pred))
        print(classification_report(y_te,pred,zero_division=0,target_names=le.classes_))

if __name__=="__main__":
    run()
