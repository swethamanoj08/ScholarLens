# 🔭 ScholarLens — Research Paper Summarization & Recommendation

**Problem statement:** Research Paper Summarization & Recommendation — Transformers, Embeddings, Ranking.

## Advanced features
- **Hybrid retrieval:** TF-IDF + dense embeddings fused with Reciprocal Rank Fusion, then cross-encoder rerank (4 selectable rankers)
- **RAG Q&A:** answers grounded in retrieved papers with [n] citations; auto literature-review paragraph
- **Prompt-engineering comparison:** DistilBART vs. 4 LLM prompt styles side by side
- **KeyBERT-style keyphrases (MMR)**, **personalised recommendations** (profile vector), **topic map** (KMeans + PCA, TF-IDF cluster labels)

## Accounts & security
Email/password sign up and login (SQLite, `scholarlens.db`). Passwords are hashed with scrypt and a per-user salt (never stored in plain text), checked against a strength policy, and accounts lock for 5 minutes after 5 failed logins. Each user's saved papers, search history and activity are stored per account and only ever queried by that user's id (parameterised SQL). Logged-in state lasts for the browser session; data persists across sessions.

## Reliability
The Research Assistant answers from the paper library first and automatically falls back to a Groq general answer when no relevant papers are found or retrieval fails.

## Rubric mapping (30 marks)
| Criterion | Where in the app |
|---|---|
| Extraction, preprocessing, vectorization (CO1) | PDF/arXiv extraction via pypdf, text cleaning and de-duplication (`build_index.py`), TF-IDF and MiniLM embeddings |
| Sentiment + information extraction (CO2) | Sentiment & IE tab: DistilBERT-SST2 sentence sentiment, BERT-NER, regex metric extraction, LLM JSON extraction |
| Topic modelling / multilingual + cloud AI (CO3) | Research Assistant (English + Hindi text/voice via Whisper on Groq; Groq translation with MarianMT fallback); Groq + Hugging Face Hub |
| GenAI/LLM + prompt engineering (CO4) | Summarize (4 prompt styles), Research Assistant (RAG with citations + Groq fallback), literature-review generator |

## Objectives
1. Retrieve relevant arXiv papers for a free-text query using semantic embeddings.
2. Improve ranking with a cross-encoder reranker; compare against a TF-IDF baseline.
3. Summarize papers (pasted text / PDF) with a local transformer and an LLM with prompt engineering.
4. Recommend similar papers from a liked paper.

## Technologies
Python, Streamlit, Sentence-Transformers (`all-MiniLM-L6-v2`), Cross-Encoder (`ms-marco-MiniLM-L-6-v2`),
HuggingFace Transformers (`distilbart-cnn-12-6`), scikit-learn (TF-IDF), Groq API (Llama 3.3 70B), pypdf.

## Architecture
```
arXiv abstracts -> clean -> TF-IDF + MiniLM embeddings (offline, build_index.py)
Query -> embed -> cosine top-50 -> cross-encoder rerank -> top-k results
Text/PDF -> extract (pypdf) -> clean -> chunk -> DistilBART  OR  prompt template -> Llama 3.3 (Groq)
```

## Methodology
- **Text extraction:** pypdf for uploaded PDFs; dataset abstracts via HF `datasets`.
- **Preprocessing:** LaTeX removal, whitespace normalisation, de-duplication.
- **Vectorization:** TF-IDF (1–2 grams, sublinear) baseline; 384-d normalised sentence embeddings.
- **Ranking:** bi-encoder retrieval then cross-encoder rerank (two-stage retrieve-and-rerank).
- **Summarization:** chunked abstractive (DistilBART) and 4 prompt styles: zero-shot, structured, ELI5, chain-of-thought.

## Dataset
`gfissore/arxiv-abstracts-2021` (HuggingFace). First 5,000 papers in cs.CL / cs.LG / cs.AI / cs.CV / cs.IR.
Fields: id, title, abstract, categories. Preprocessing in `build_index.py`.

## Install & run
```bash
pip install -r requirements.txt
python build_index.py            # creates papers.csv, embeddings.npy, tfidf.joblib (~5-10 min)
export GROQ_API_KEY=your_key     # optional, enables LLM summaries
streamlit run app.py
```

## Deploy (live URL)
Push the repo **including** `papers.csv`, `embeddings.npy`, `tfidf.joblib` to GitHub, then on
[share.streamlit.io](https://share.streamlit.io) pick the repo and `app.py`; add `GROQ_API_KEY` under Secrets.
(Alternative: Hugging Face Spaces, Streamlit SDK.)

## Project structure
`build_index.py` · `app.py` · `requirements.txt` · `papers.csv` · `embeddings.npy` · `tfidf.joblib`

## Author
Swetha — B.Tech AIML, MITAOE