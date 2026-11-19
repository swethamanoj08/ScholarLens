"""Step 1: download arXiv abstracts, preprocess, vectorize (TF-IDF + transformer embeddings)."""
import re, joblib, numpy as np, pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

N = 5000
KEEP = ("cs.CL", "cs.LG", "cs.AI", "cs.CV", "cs.IR")

def clean(t: str) -> str:
    t = re.sub(r"\$[^$]*\$", " ", t)          # strip inline LaTeX
    t = re.sub(r"\s+", " ", t).strip()
    return t

ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
rows = []
for r in ds:
    cats = r["categories"]
    cats = cats.split() if isinstance(cats, str) else (cats or [])
    if not any(c in KEEP for c in cats):
        continue
    rows.append({"id": r["id"], "title": clean(r["title"]),
                 "abstract": clean(r["abstract"]), "categories": " ".join(cats)})
    if len(rows) >= N:
        break

df = pd.DataFrame(rows).drop_duplicates("id")
df["text"] = df["title"] + ". " + df["abstract"]
df.to_csv("papers.csv", index=False)

tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
joblib.dump((tfidf, tfidf.fit_transform(df["text"])), "tfidf.joblib")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(df["text"].tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=True)
np.save("embeddings.npy", emb.astype("float32"))
print("Indexed", len(df), "papers; embeddings", emb.shape)
