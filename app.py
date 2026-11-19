import os, re, html, joblib, numpy as np, pandas as pd, streamlit as st, plotly.express as px
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import auth_db as adb

st.set_page_config(page_title="ScholarLens", page_icon="🔭", layout="wide")

import plotly.io as pio, plotly.graph_objects as go
_t = go.layout.Template(pio.templates["plotly_dark"])
_t.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e6e8ec", size=12),
    xaxis=dict(gridcolor="#343840", linecolor="#454b55", zerolinecolor="#454b55"), yaxis=dict(gridcolor="#343840", linecolor="#454b55", zerolinecolor="#454b55"),
    colorway=["#6c9bd8", "#8d9db6", "#a9c1e6", "#c0c6d1", "#d1a85c", "#5fbf95", "#d08a82", "#a79fd0", "#7fc0cb", "#b2b9c4", "#d9b48a", "#8396b3"])
pio.templates["scholar"] = _t
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:wght@500;600&display=swap');
.stApp,.stApp p,.stApp label,.stApp li,.stApp button,.stApp input,.stApp textarea{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif}
.stApp{background:#1f2125;color:#e6e8ec}
[data-testid='stHeader']{background:transparent}
.block-container{max-width:1120px;padding:1.4rem 2rem 4rem}
h1,h2,h3,h4{font-family:'Source Serif 4',Georgia,serif!important;color:#f1f3f6;font-weight:600;letter-spacing:-.01em}
h3{font-size:1.3rem!important}
hr{border-color:#383c44!important;margin:1.2rem 0!important}
[data-testid='stCaptionContainer']{color:#9aa1ad}
a{color:#8db4e8}
/* top bar */
.topbar{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;padding:2px 0 12px}
.brand{font-family:'Source Serif 4',Georgia,serif;font-size:1.6rem;font-weight:600;color:#f1f3f6}
.tag{margin-left:12px;color:#9aa1ad;font-size:.85rem}
.stats{display:flex;gap:18px;font-size:.76rem;color:#7d8491}.stats b{color:#b4bac5;font-weight:600}
/* navigation */
.st-key-nav{margin-bottom:1.2rem}
.st-key-nav [role='radiogroup']{gap:2px;border-bottom:1px solid #383c44;flex-wrap:wrap}
.st-key-nav label[data-baseweb='radio']{margin:0;padding:10px 16px;border-bottom:2px solid transparent;border-radius:6px 6px 0 0;cursor:pointer;color:#9aa1ad;transition:.12s}
.st-key-nav label[data-baseweb='radio']>div:first-child{display:none}
.st-key-nav label[data-baseweb='radio']:hover{background:#2a2d33;color:#e6e8ec}
.st-key-nav label[data-baseweb='radio']:has(input:checked){border-bottom-color:#6c9bd8;color:#fff;font-weight:600;background:transparent}
.st-key-nav label[data-baseweb='radio'] p{font-size:.92rem}
/* buttons */
[data-testid^='stBaseButton'],[data-testid^='stBaseLinkButton']{border-radius:8px;font-weight:500;font-size:.87rem;padding:.38rem 1rem;border:1px solid #454b55;background:#2a2d33;color:#e6e8ec;box-shadow:0 1px 1px rgba(0,0,0,.25);transition:.12s}
[data-testid^='stBaseButton']:hover,[data-testid^='stBaseLinkButton']:hover{border-color:#6c9bd8;color:#9cc0f0;background:#2f333a}
[data-testid='stBaseButton-primary']{background:#4a7bbf;border-color:#4a7bbf;color:#fff}
[data-testid='stBaseButton-primary']:hover{background:#5a8bcf;border-color:#5a8bcf;color:#fff}
/* inputs, containers */
div[data-baseweb='input'],div[data-baseweb='textarea'],div[data-baseweb='select']>div{border-radius:8px!important;background:#2a2d33!important;border-color:#424851!important}
[data-testid='stExpander']{background:#262a30;border:1px solid #383c44!important;border-radius:10px}
[data-testid='stAlert']{border-radius:10px}
[data-testid='stMetric']{background:#262a30;border:1px solid #383c44;border-radius:10px;padding:10px 14px}
[data-testid='stMetricLabel']{font-size:.72rem!important;color:#9aa1ad}
[data-testid='stMetricValue']{font-size:1.25rem!important;font-weight:600;color:#f1f3f6}
button[data-baseweb='tab']{font-size:.88rem}
[data-baseweb='tab-highlight']{background:#6c9bd8!important}
/* paper cards, chips, sentiment cards */
.card{background:#262a30;border:1px solid #383c44;border-radius:10px;padding:16px 18px;margin:10px 0;box-shadow:0 1px 2px rgba(0,0,0,.25);transition:border-color .15s}
.card:hover{border-color:#5d82b3}
.card h4{margin:0 0 6px;font-size:1.05rem}
.card p{color:#b4bac5;margin:8px 0;font-size:.92rem;line-height:1.55}
.card small{color:#8a919e}
.chip{display:inline-block;background:#353a42;color:#c5cad3;border-radius:6px;padding:2px 9px;margin:0 6px 4px 0;font-size:.74rem;font-weight:500}
.rk{float:right;color:#6f7683;font-weight:600;font-size:.95rem}
.sc{background:#262a30;border:1px solid #383c44;border-left:3px solid;border-radius:8px;padding:9px 14px;margin:6px 0;font-size:.92rem;line-height:1.5}
.sc.pos{border-left-color:#4fae85}.sc.neg{border-left-color:#d0766c}.sc small{color:#8a919e}
[data-testid='stForm']{background:#262a30;border:1px solid #383c44;border-radius:12px;padding:1.1rem}
</style>""", unsafe_allow_html=True)

# ================= Authentication gate (SQLite, scrypt) =================
@st.cache_resource
def _init_db(): adb.init_db(); return True
_init_db()

def auth_screen():
    st.markdown('<div class="topbar"><div><span class="brand">ScholarLens</span><span class="tag">Research paper search, summarization &amp; recommendation</span></div></div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown("### Welcome"); st.caption("Sign in to keep your saved papers, search history and activity across sessions.")
        t1, t2 = st.tabs(["Log in", "Sign up"])
        with t1:
            with st.form("login_form"):
                e = st.text_input("Email", key="li_email"); pw = st.text_input("Password", type="password", key="li_pw")
                sub = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if sub:
                u, msg = adb.authenticate(e, pw)
                if u: adb.log_activity(u["id"], "login"); st.session_state["user"] = u; st.rerun()
                else: st.error(msg)
        with t2:
            with st.form("signup_form"):
                nm = st.text_input("Full name", key="su_name"); em = st.text_input("Email", key="su_email")
                p1 = st.text_input("Password", type="password", key="su_p1"); p2 = st.text_input("Confirm password", type="password", key="su_p2")
                st.caption("At least 8 characters, with upper and lower case letters, a number and a special character.")
                sub2 = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if sub2:
                if p1 != p2: st.error("Passwords do not match.")
                else:
                    ok, msg = adb.create_user(em, nm, p1)
                    if ok:
                        u, _ = adb.authenticate(em, p1); adb.log_activity(u["id"], "signup"); st.session_state["user"] = u; st.rerun()
                    else: st.error(msg)

user = st.session_state.get("user")
if not user: auth_screen(); st.stop()                    # nothing below (models, app) loads until the user is logged in
uid = user["id"]; SAVED = adb.saved_ids(uid)


@st.cache_resource
def load_all():
    df = pd.read_csv("papers.csv", dtype={"id": str}); tfidf, mat = joblib.load("tfidf.joblib")
    return (df, np.load("embeddings.npy"), tfidf, mat,
            SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
            CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"))
df, emb, tfidf, mat, bi, ce = load_all()

@st.cache_data(show_spinner=False)
def topics(n=12):
    lab = KMeans(n, n_init=5, random_state=0).fit_predict(emb)
    terms = np.array(tfidf.get_feature_names_out())
    names = {c: ", ".join(terms[np.asarray(mat[lab == c].mean(axis=0)).ravel().argsort()[::-1][:3]]) for c in range(n)}
    return lab, names, PCA(2, random_state=0).fit_transform(emb)

def pre(t): return re.sub(r"\s+", " ", t).strip()
def _r(s): r = np.empty(len(s)); r[s.argsort()[::-1]] = np.arange(len(s)); return r

MODES = ["Hybrid + Rerank", "Hybrid (RRF)", "Dense", "TF-IDF"]
@st.cache_data(show_spinner=False)
def rank(q, k=10, mode=MODES[0]):
    q = pre(q); dense = emb @ bi.encode([q], normalize_embeddings=True)[0]
    lex = linear_kernel(tfidf.transform([q]), mat).ravel()
    fused = 1 / (60 + _r(dense)) + 1 / (60 + _r(lex))        # Reciprocal Rank Fusion
    if mode == "Hybrid + Rerank":
        cand = fused.argsort()[::-1][:30]
        s = ce.predict([(q, df.text.iloc[i]) for i in cand]); o = np.argsort(s)[::-1][:k]
        return df.iloc[cand[o]].assign(score=s[o])
    s = {"Hybrid (RRF)": fused, "Dense": dense, "TF-IDF": lex}[mode]
    idx = s.argsort()[::-1][:k]; return df.iloc[idx].assign(score=s[idx])

@st.cache_data(show_spinner=False)
def keyphrases(text, n=6, div=0.6):                            # KeyBERT-style MMR
    try: cands = list(CountVectorizer(ngram_range=(1, 3), stop_words="english").fit([text]).get_feature_names_out())[:200]
    except ValueError: return []
    d = bi.encode([text], normalize_embeddings=True)[0]; c = bi.encode(cands, normalize_embeddings=True)
    rel = c @ d; sel = [int(rel.argmax())]
    while len(sel) < min(n, len(cands)):
        sc = (1 - div) * rel - div * (c @ c[sel].T).max(1); sc[sel] = -9; sel.append(int(sc.argmax()))
    return [cands[i] for i in sel]

@st.cache_data(show_spinner=False)
def pick_model(key):
    from groq import Groq
    ids = [m.id for m in Groq(api_key=key).models.list().data]
    for m in ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]:
        if m in ids: return m
    skip = ("whisper", "tts", "guard", "orpheus", "compound", "safeguard")
    return [i for i in ids if not any(x in i for x in skip)][0]

def llm(prompt):
    from groq import Groq
    try: key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
    except Exception: key = ""
    if not key: return "⚠️ Set GROQ_API_KEY (env var or Streamlit secrets) to enable LLM features."
    m = pick_model(key)
    out = Groq(api_key=key).chat.completions.create(model=m, temperature=0.2,
        messages=[{"role": "user", "content": prompt[:14000]}]).choices[0].message.content
    out = re.sub(r"<br\s*/?>", " · ", out)
    return out + f"\n\n*Model: `{m}`*"

@st.cache_resource
def local_sum():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    n = "sshleifer/distilbart-cnn-12-6"
    return AutoTokenizer.from_pretrained(n), AutoModelForSeq2SeqLM.from_pretrained(n)
def summarize_local(t):
    import torch
    tok, mod = local_sum()
    w = t.split(); ch = [" ".join(w[i:i + 600]) for i in range(0, len(w), 600)][:4]
    out = []
    for c in ch:
        x = tok(c, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            ids = mod.generate(**x, max_length=130, min_length=40, num_beams=2, early_stopping=True)
        out.append(tok.decode(ids[0], skip_special_tokens=True))
    return " ".join(out)

PROMPTS = {"Zero-shot": "Summarize this research paper in 5 sentences.\n\n{t}",
 "Structured (role + format)": "You are an expert research assistant. Summarize as **Problem**, **Method**, **Key Results**, **Limitations** (1-2 lines each).\n\n{t}",
 "ELI5": "Explain this paper to a first-year student with a simple analogy, under 120 words.\n\n{t}",
 "Chain-of-thought": "First list the main claims, then judge which are best supported, then give a final 3-sentence summary.\n\n{t}"}

import urllib.request, urllib.parse, xml.etree.ElementTree as ET
_ATOM, _ARX = "{http://www.w3.org/2005/Atom}", "{http://arxiv.org/schemas/atom}"

def aq(text):                                            # plain text -> arXiv query
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
    return " AND ".join(f"all:{w}" for w in [w for w in re.findall(r"\w+", text.lower()) if w not in SW][:8])

@st.cache_data(ttl=3600, show_spinner=False)
def arxiv_fetch(query=None, ids=None, n=20):             # LIVE arXiv API (papers outside the 5,000)
    p = {"id_list": ",".join(ids), "max_results": len(ids)} if ids else {"search_query": query, "max_results": n, "sortBy": "relevance"}
    req = urllib.request.Request("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(p), headers={"User-Agent": "ScholarLens/1.0"})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=25).read()); out = []
    for e in root.findall(_ATOM + "entry"):
        t = re.sub(r"\s+", " ", e.find(_ATOM + "title").text or "").strip()
        if t.lower() == "error": continue
        ab = re.sub(r"\s+", " ", e.find(_ATOM + "summary").text or "").strip(); c = e.find(_ARX + "primary_category")
        out.append({"id": e.find(_ATOM + "id").text.split("/abs/")[-1], "title": t, "abstract": ab,
                    "categories": c.get("term") if c is not None else "arXiv", "text": t + ". " + ab})
    if not out: raise ValueError("arXiv returned no papers for this request")
    return out

def rank_live(papers, vec, k=6):
    if not papers: return pd.DataFrame()
    s = bi.encode([p["text"] for p in papers], normalize_embeddings=True) @ vec; o = s.argsort()[::-1][:k]
    return pd.DataFrame([papers[i] for i in o]).assign(score=s[o])

def strip_refs(t):                                       # drop the reference list so it doesn't pollute summaries/keyphrases
    m = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", t, re.I))
    return t[:m[-1].start()] if m and m[-1].start() > len(t) * 0.4 else t

@st.cache_data(ttl=3600, show_spinner=False)
def arxiv_pdf_text(aid):
    import io; from pypdf import PdfReader
    req = urllib.request.Request(f"https://arxiv.org/pdf/{aid}", headers={"User-Agent": "ScholarLens/1.0"})
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(urllib.request.urlopen(req, timeout=90).read())).pages)

def save_btn(r, key):
    pid = str(r.id)
    if pid in SAVED: st.button("✓ Saved", key=f"sv_{key}_{pid}", disabled=True)
    else: st.button("☆ Save to library", key=f"sv_{key}_{pid}", on_click=lambda: (adb.save_paper(uid, {"id": pid, "title": r.title, "abstract": r.abstract, "categories": r.categories}), adb.log_activity(uid, "save_paper", r.title)))

def cards(res, key):
    for i, (_, r) in enumerate(res.iterrows(), 1):
        chips = "".join(f'<span class="chip">{c}</span>' for c in r.categories.split()[:3])
        st.markdown(f'<div class="card"><span class="rk">#{i}</span><h4>{html.escape(r.title)}</h4>{chips}'
                    f'<p>{html.escape(r.abstract[:300])}…</p><small>arXiv:{r.id} · score {r.score:.3f}</small></div>', unsafe_allow_html=True)
        with st.expander("Full abstract · keyphrases · link"):
            st.write(r.abstract); st.markdown("**Keyphrases:** " + " · ".join(f"`{k}`" for k in keyphrases(r.abstract)))
            st.markdown(f"[Open on arXiv](https://arxiv.org/abs/{r.id})"); save_btn(r, key)

@st.cache_data(show_spinner=False)
def evaluate(nc=1000, nq=200):
    idx = np.random.RandomState(0).choice(len(df), nc, replace=False)
    ab, ti = df.abstract.values[idx], df.title.values[idx][:nq]       # query = title, doc = abstract only
    tf = TfidfVectorizer(stop_words="english", sublinear_tf=True); M = tf.fit_transform(ab)
    E = bi.encode(list(ab), batch_size=64, normalize_embeddings=True)
    D = bi.encode(list(ti), normalize_embeddings=True) @ E.T; L = linear_kernel(tf.transform(ti), M)
    res = {m: [] for m in ["TF-IDF", "Dense", "Hybrid (RRF)", "Hybrid + Rerank"]}
    for i in range(nq):
        f = 1 / (60 + _r(D[i])) + 1 / (60 + _r(L[i]))
        for m, s in [("TF-IDF", L[i]), ("Dense", D[i]), ("Hybrid (RRF)", f)]: res[m].append((s > s[i]).sum() + 1)
        cand = list(f.argsort()[::-1][:20])
        if i in cand:
            sc = ce.predict([(ti[i], ab[j]) for j in cand]); res["Hybrid + Rerank"].append((sc > sc[cand.index(i)]).sum() + 1)
        else: res["Hybrid + Rerank"].append(10**6)
    rows = []
    for m, r in res.items():
        r = np.array(r); rows.append({"Method": m, "MRR@10": np.where(r <= 10, 1 / r, 0).mean(),
            "Recall@10": (r <= 10).mean(), "nDCG@10": np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()})
    return pd.DataFrame(rows)

st.markdown(f'<div class="topbar"><div><span class="brand">ScholarLens</span><span class="tag">Research paper search, summarization &amp; recommendation</span></div>'
            f'<div class="stats"><span><b>{len(df):,}</b> papers</span><span><b>{emb.shape[1]}</b>-d embeddings</span><span><b>3</b>-stage retrieval</span><span>Groq LLM</span></div></div>', unsafe_allow_html=True)

u1, u2 = st.columns([7, 1.3]); u1.caption(f"Signed in as **{user['name']}** · {user['email']}")
if u2.button("Log out", key="logout", use_container_width=True): adb.log_activity(uid, "logout"); st.session_state.clear(); st.rerun()

NAMES = ['🔎 Search', '🧠 Summarize', '🤖 Research Assistant', '⭐ For You', '🗺️ Topic Map', '😊 Sentiment & IE', '📁 My Library']
page = st.radio("Section", NAMES, horizontal=True, label_visibility="collapsed", key="nav", format_func=lambda n: n.split(" ", 1)[1])

if page == NAMES[0]:
    q = st.text_input("What do you want to read about?", "graph neural networks for molecule property prediction")
    c1, c2 = st.columns([3, 1]); mode = c1.radio("Ranker", MODES, horizontal=True); k = c2.slider("Results", 3, 20, 8)
    if q:
        if st.session_state.get("lq") != q: adb.log_search(uid, "search", q); st.session_state["lq"] = q
        res = rank(q, k, mode); cards(res, "s")
        if st.checkbox("🌐 Also search live arXiv (papers outside the 5,000)"):
            try:
                st.subheader("Live from arXiv")
                cards(rank_live(arxiv_fetch(query=aq(q), n=25), bi.encode([q], normalize_embeddings=True)[0]), "live")
            except Exception as e: st.error(f"arXiv lookup failed: {e}")
        if st.button("✍️ Generate literature-review paragraph"):
            src = "\n".join(f"[{i}] {r.title}: {r.abstract[:500]}" for i, (_, r) in enumerate(res.head(5).iterrows(), 1))
            st.info(llm(f"Write a 150-word related-work paragraph on '{q}' using only these papers, citing as [n].\n\n{src}"))

if page == NAMES[1]:
    src = st.radio("Input", ["Paste text", "Upload PDF", "arXiv link / ID", "Indexed paper"], horizontal=True); text = ""
    if src == "Paste text": text = st.text_area("Paper text", height=180)
    elif src == "Upload PDF":
        f = st.file_uploader("PDF", type="pdf")
        if f:
            from pypdf import PdfReader
            text = strip_refs("\n".join(p.extract_text() or "" for p in PdfReader(f).pages))
    elif src == "arXiv link / ID":
        link = st.text_input("arXiv link or ID", placeholder="https://arxiv.org/abs/1706.03762")
        m = re.search(r"\d{4}\.\d{4,5}(v\d+)?", link)
        if m:
            try:
                with st.spinner("Downloading PDF from arXiv..."): text = strip_refs(arxiv_pdf_text(m.group(0)))
                meta = arxiv_fetch(ids=(m.group(0).split("v")[0],))
                st.caption(f"**{meta[0]['title'] if meta else m.group(0)}** · {len(text.split()):,} words extracted (references removed)")
            except Exception as e: st.error(f"Could not fetch this paper: {e}")
        elif link: st.warning("No arXiv ID found. Use a link like https://arxiv.org/abs/1706.03762 or an ID like 1706.03762")
    else: text = df.abstract.iloc[df.title.tolist().index(st.selectbox("Paper", df.title[:500]))]
    text = pre(text); styles = st.multiselect("Prompt styles to compare", list(PROMPTS), ["Structured (role + format)"])
    use_local = st.checkbox("Also run local DistilBART (slow on CPU, downloads ~1.2 GB first time)", True)
    if st.button("Summarize") and text:
        adb.log_activity(uid, "summarize", text[:80])
        cols = st.columns(1 + len(styles))
        for col, sty in zip(cols[1:], styles):               # LLM first, so it shows even if local is slow
            with col:
                st.subheader(sty)
                with st.spinner("Asking the LLM..."): st.write(llm(PROMPTS[sty].format(t=text)))
        st.markdown("**Keyphrases (MMR):** " + " · ".join(f"`{k}`" for k in keyphrases(text)))
        if use_local:
            with cols[0]:
                st.subheader("DistilBART")
                with st.spinner("Loading model & summarizing (first run is slow)..."):
                    s = summarize_local(text)
                st.write(s); st.caption(f"{len(text.split())} → {len(s.split())} words")

if page == NAMES[3]:
    ext = st.session_state.setdefault("ext", [])
    liked = st.multiselect("Pick papers you liked (type to search all 5,000 titles)", df.title.tolist())
    with st.expander("➕ Add papers from anywhere (arXiv links / IDs, or live arXiv search)", expanded=True):
        links = st.text_area("Paste arXiv links or IDs, one per line", placeholder="https://arxiv.org/abs/1706.03762")
        qs = st.text_input("…or search arXiv live", placeholder="e.g. vision transformers for medical images")
        c1, c2, c3 = st.columns(3)
        try:
            if c1.button("Add from links") and links:
                fid = tuple(dict.fromkeys(re.findall(r"\d{4}\.\d{4,5}", links)))
                if not fid: st.warning("No arXiv IDs found. Use links like https://arxiv.org/abs/1706.03762")
                else:
                    have = {e["id"] for e in ext}; new = [p for p in arxiv_fetch(ids=fid) if p["id"] not in have]
                    ext += new; st.success(f"Added {len(new)} paper(s): " + "; ".join(p["title"][:60] for p in new))
            if c2.button("Search arXiv live") and qs:
                with st.spinner("Searching arXiv..."): st.session_state["found"] = arxiv_fetch(query=aq(qs), n=8)
                st.success(f"Found {len(st.session_state['found'])} papers. Tick the ones you like below.")
        except Exception as e: st.error(f"arXiv lookup failed: {e}")
        if c3.button("Clear added papers"): ext.clear(); st.session_state["found"] = []
        sel = [p for p in st.session_state.get("found", []) if st.checkbox(f"{p['title']}  (arXiv:{p['id']})", key="f" + p["id"])]
    mine = ext + [p for p in sel if p["id"] not in {e["id"] for e in ext}]
    if mine: st.caption("Added from arXiv: " + " · ".join(p["title"][:50] for p in mine))
    ids = [df.title.tolist().index(t) for t in liked]; vecs = ([emb[ids]] if ids else [])
    if mine: vecs.append(bi.encode([p["text"] for p in mine], normalize_embeddings=True))
    if vecs:
        v = np.vstack(vecs).mean(0); v /= np.linalg.norm(v); s = emb @ v; s[ids] = -1
        top = s.argsort()[::-1][:8]; st.subheader("📚 From the 5,000-paper library"); cards(df.iloc[top].assign(score=s[top]), "u")
        st.subheader("🌐 Fresh from arXiv (any paper, not limited to the library)")
        if st.button("Fetch live recommendations"):
            seeds = [df.abstract.iloc[i] for i in ids[:3]] + [p["abstract"] for p in mine[:3]]
            ph = [x for t in seeds for x in keyphrases(t, 3)[:2]]
            try:
                known = {p["id"] for p in mine}; live = [p for p in arxiv_fetch(query=" OR ".join(f'all:"{x}"' for x in ph[:6]), n=40) if p["id"] not in known]
                cards(rank_live(live, v, 8), "lv")
            except Exception as e: st.error(f"arXiv lookup failed: {e}")
    else: st.info("Pick a library paper, or add one from arXiv above, to get recommendations.")

def chips(items): return " ".join(f'<span class="chip">{html.escape(str(i))}</span>' for i in items)

if page == NAMES[4]:   # ---- Topic Map: interactive research discovery
    lab, names, xy = topics(); n = len(names)
    lbl = {c: (names[c] if list(names.values()).count(names[c]) == 1 else f"{names[c]} #{c}") for c in names}
    p = df.assign(x=xy[:, 0], y=xy[:, 1], topic=[lbl[l] for l in lab], idx=np.arange(len(df)))
    st.caption("Click any dot to jump to its topic and paper, or pick a topic below.")
    ev = st.plotly_chart(px.scatter(p, x="x", y="y", color="topic", hover_name="title", custom_data=["idx"], height=560, template="scholar"),
                         use_container_width=True, theme=None, on_select="rerun", selection_mode="points", key="tm_scatter")
    st.plotly_chart(px.bar(p.topic.value_counts().reset_index(), x="count", y="topic", orientation="h", template="scholar"), use_container_width=True, theme=None)

    try: pts = ev.selection.points
    except Exception: pts = []
    def clicked_idx(pt):                                     # robust: customdata may be a list, a dict, or missing
        try:
            cd = pt.get("customdata")
            return int(list(cd.values())[0] if isinstance(cd, dict) else (cd[0] if isinstance(cd, (list, tuple)) else cd))
        except Exception: pass
        try:                                                 # fallback: trace (topic) + position inside that trace
            rows = np.where(p.topic.values == list(p.topic.unique())[pt["curve_number"]])[0]
            return int(rows[pt.get("point_index", pt.get("point_number"))])
        except Exception: return None
    if pts:                                                  # a dot was clicked -> open its topic and paper
        ci = clicked_idx(pts[0])
        if ci is not None and 0 <= ci < len(df) and st.session_state.get("tm_last") != ci:
            st.session_state.update(tm_last=ci, tm_topic=lbl[lab[ci]], tm_sel=ci)
    order = list(p.topic.value_counts().index)
    if st.session_state.get("tm_topic") not in order: st.session_state["tm_topic"] = order[0]
    topic = st.selectbox("🔍 Explore a topic", order, key="tm_topic")

    c = [k for k in lbl if lbl[k] == topic][0]; mem = np.where(lab == c)[0]
    cent = np.vstack([emb[lab == k].mean(0) for k in range(n)]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    mem = mem[np.argsort(-(emb[mem] @ cent[c]))]               # most representative papers first
    terms = np.array(tfidf.get_feature_names_out()); kw = terms[np.asarray(mat[lab == c].mean(axis=0)).ravel().argsort()[::-1][:10]]
    m1, m2, m3 = st.columns(3); m1.metric("Papers in topic", len(mem)); m2.metric("Share of library", f"{len(mem) / len(df):.0%}")
    m3.metric("Top category", pd.Series(" ".join(df.categories.iloc[mem]).split()).value_counts().index[0])
    st.markdown("**Topic keywords:** " + chips(kw), unsafe_allow_html=True)
    st.markdown("**Related topics** (nearest by meaning)")
    for col, k in zip(st.columns(3), np.argsort(-(cent @ cent[c]))[1:4]):
        col.button(f"{lbl[k]}  ·  {cent[k] @ cent[c]:.0%} similar", key=f"rel{k}", on_click=lambda t=lbl[k]: st.session_state.update(tm_topic=t))

    sums = st.session_state.setdefault("tm_sum", {}); sel = st.session_state.get("tm_sel")
    if sel is not None:                                      # selected paper panel
        r = df.iloc[sel]; st.markdown("---"); st.subheader("📄 " + r.title)
        st.markdown(chips(r.categories.split()[:4]) + f" &nbsp;topic: *{lbl[lab[sel]]}*", unsafe_allow_html=True)
        st.write(r.abstract); st.markdown("**Keywords:** " + chips(keyphrases(r.abstract)), unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4); b1.link_button("Open on arXiv ↗", f"https://arxiv.org/abs/{r.id}")
        go = b2.button("🧠 Summarize with LLM", key="tm_sumbtn"); b3.button("✖ Clear selection", on_click=lambda: st.session_state.update(tm_sel=None))
        with b4: save_btn(r, "tmsel")
        if (go or st.session_state.pop("tm_do", False)) and sel not in sums:
            with st.spinner("Summarizing..."): out = llm(PROMPTS["Structured (role + format)"].format(t=r.abstract))
            if out.startswith("⚠️"): st.warning(out)
            else: sums[sel] = out
        if sel in sums: st.markdown(sums[sel])
        if st.checkbox("Show similar papers", key="tm_simchk"):
            sm = emb @ emb[sel]; sm[sel] = -1; top = sm.argsort()[::-1][:5]; cards(df.iloc[top].assign(score=sm[top]), "tm_sim")

    st.markdown("---"); st.subheader(f"Papers in: {topic}")
    f1, f2, f3 = st.columns([3, 1, 1])
    q = f1.text_input("Filter within this topic", placeholder="keyword in title or abstract", key="tm_q")
    nshow = f2.slider("Show", 5, 30, 10, key="tm_n"); sort = f3.selectbox("Sort by", ["Most representative", "A–Z"], key="tm_sort")
    sub = [i for i in mem if q.lower() in (df.title.iloc[i] + " " + df.abstract.iloc[i]).lower()]
    if sort == "A–Z": sub.sort(key=lambda i: df.title.iloc[i])
    st.caption(f"{len(sub)} papers match · showing {min(nshow, len(sub))}")
    for i in sub[:nshow]:
        r = df.iloc[i]
        with st.expander(r.title):
            st.caption(f"arXiv:{r.id} · {r.categories}"); st.write(r.abstract)
            st.markdown("**Keywords:** " + chips(keyphrases(r.abstract)), unsafe_allow_html=True)
            x1, x2, x3 = st.columns(3); x1.link_button("Open on arXiv ↗", f"https://arxiv.org/abs/{r.id}")
            x2.button("🧠 Select & summarize", key=f"pick{i}", on_click=lambda i=i: st.session_state.update(tm_sel=i, tm_do=True))
            with x3: save_btn(r, f"tm{i}")

# ================= Advanced NLP modules (Hugging Face) =================
import json, torch
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

@st.cache_resource
def hf_model(name, kind):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
    M = AutoModelForSequenceClassification if kind == "cls" else AutoModelForTokenClassification
    return AutoTokenizer.from_pretrained(name), M.from_pretrained(name).eval()

def sent_split(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) > 4]

def sentiment(sents):                                   # DistilBERT-SST2
    tok, m = hf_model("distilbert-base-uncased-finetuned-sst-2-english", "cls"); out = []
    for i in range(0, len(sents), 16):
        x = tok(sents[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad(): p = torch.softmax(m(**x).logits, -1)
        out += [(m.config.id2label[int(j)], float(p[k, j])) for k, j in enumerate(p.argmax(-1))]
    return out

def ner(text):                                          # BERT-NER, BIO merge
    tok, m = hf_model("dslim/bert-base-NER", "tok")
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
    off = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad(): ids = m(**enc).logits.argmax(-1)[0].tolist()
    ents, cur = [], None
    for (a, b), i in zip(off, ids):
        if a == b: continue
        lab = m.config.id2label[i]
        if lab == "O": cur = None; continue
        if lab.startswith("B-") or cur is None or cur[0] != lab[2:]: cur = [lab[2:], a, b]; ents.append(cur)
        else: cur[2] = b
    return pd.DataFrame([(t, text[a:b]) for t, a, b in ents], columns=["type", "entity"]).drop_duplicates()

METRIC = re.compile(r"\b(accuracy|F1(?:-score)?|BLEU|ROUGE-?\w*|precision|recall|AUC|mAP|perplexity|MRR|nDCG)\b[^.\d]{0,40}?(\d+(?:\.\d+)?\s?%?)", re.I)
def metrics(text): return pd.DataFrame([(a, b) for a, b in METRIC.findall(text)], columns=["metric", "value"]).drop_duplicates()

@st.cache_resource
def lemma_fn():
    try:
        import nltk; nltk.download("wordnet", quiet=True); w = WordNetLemmatizer(); w.lemmatize("tests"); return w.lemmatize
    except Exception: return PorterStemmer().stem

@st.cache_data(show_spinner=False)
def lda_model(k=8):
    cv = CountVectorizer(stop_words="english", max_df=.5, min_df=5, max_features=5000); X = cv.fit_transform(df.abstract)
    lda = LatentDirichletAllocation(k, random_state=0, max_iter=15, learning_method="online").fit(X)
    terms = cv.get_feature_names_out(); tops = [terms[c.argsort()[::-1][:10]] for c in lda.components_]
    B = (X > 0).astype(int).tocsc(); D = np.asarray(B.sum(0)).ravel(); ix = {w: i for i, w in enumerate(terms)}; coh = []
    for t in tops:                                       # UMass coherence
        ii = [ix[w] for w in t]
        coh.append(sum(np.log((B[:, ii[a]].multiply(B[:, ii[b]]).sum() + 1) / D[ii[b]]) for a in range(1, 10) for b in range(a)))
    return lda.transform(X), tops, float(lda.perplexity(X)), float(np.mean(coh))

LANG = {"Hindi": "hi", "French": "fr", "German": "de", "Spanish": "es"}
@st.cache_resource
def mt(src, tgt):
    from transformers import MarianMTModel, MarianTokenizer
    n = f"Helsinki-NLP/opus-mt-{src}-{tgt}"; return MarianTokenizer.from_pretrained(n), MarianMTModel.from_pretrained(n).eval()
def translate(texts, src, tgt):
    tok, m = mt(src, tgt); x = tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad(): ids = m.generate(**x, num_beams=4, max_length=256)
    return [tok.decode(i, skip_special_tokens=True) for i in ids]

def pick_paper(key):
    t = st.selectbox("Paper", df.title[:500], key=key); return df.iloc[df.title.tolist().index(t)]

@st.cache_data(show_spinner=False)
def ie_analyze(text):                                    # DistilBERT-SST2 + BERT-NER + regex, computed once per text
    ss = sent_split(text)[:60]; res = sentiment(ss) if ss else []
    return (pd.DataFrame({"sentence": ss, "label": [a for a, _ in res], "confidence": [b for _, b in res]}), ner(text), metrics(text))

def chips(items): return " ".join(f'<span class="chip">{html.escape(str(i))}</span>' for i in items)

if page == NAMES[5]:   # ---- Sentiment & Information Extraction dashboard
    r = pick_paper("p2")
    with st.expander("Abstract / analyze your own text instead"):
        st.write(r.abstract); custom = st.text_area("Paste any abstract or paragraph (optional)", height=110)
    text = pre(custom) if custom.strip() else r.abstract
    if st.button("▶ Analyze", type="primary"):
        adb.log_activity(uid, "analyze", r.title if not custom.strip() else "custom text")
        with st.spinner("Running sentiment, entity, metric and LLM extraction (first run loads models)..."):
            ie_analyze(text)
            raw = llm("Return ONLY valid JSON with keys task, methods (list), datasets (list), metrics (list), key_findings (list) for this text:\n" + text)
            st.session_state["ie"] = {"text": text, "llm": raw.split("\n\n*Model")[0]}
    ie = st.session_state.get("ie")
    if ie and ie["text"] == text:
        sdf, ents, mets = ie_analyze(text); n = len(sdf); pos = int((sdf.label == "POSITIVE").sum())
        c = st.columns(5)
        c[0].metric("Sentences", n); c[1].metric("Positive", f"{pos / n:.0%}" if n else "–")
        c[2].metric("Negative", f"{(n - pos) / n:.0%}" if n else "–"); c[3].metric("Entities", len(ents)); c[4].metric("Metrics found", len(mets))
        t1, t2, t3, t4 = st.tabs(["😊 Sentiment", "🏷️ Entities", "📏 Metrics", "🤖 LLM extraction"])
        with t1:
            f1, f2 = st.columns([2, 1])
            view = f1.radio("Show", ["All", "Positive", "Negative"], horizontal=True, key="ie_f")
            conf = f2.slider("Min. confidence", 0.5, 0.99, 0.5, key="ie_c")
            sub = sdf[sdf.confidence >= conf]
            if view != "All": sub = sub[sub.label == view.upper()]
            st.caption(f"Showing {len(sub)} of {n} sentences · negative sentences usually point to problems or limitations the paper addresses")
            for _, x in sub.iterrows():
                st.markdown(f'<div class="sc {"pos" if x.label == "POSITIVE" else "neg"}">{html.escape(x.sentence)}<br>'
                            f'<small>{x.label.title()} · {x.confidence:.0%}</small></div>', unsafe_allow_html=True)
        with t2:
            names = {"PER": "People", "ORG": "Organizations", "LOC": "Locations", "MISC": "Other"}
            if ents.empty: st.info("No named entities found in this text.")
            for t, g in ents.groupby("type"):
                st.markdown(f"**{names.get(t, t)}** ({len(g)})"); st.markdown(chips(g.entity), unsafe_allow_html=True)
        with t3:
            if mets.empty: st.info("No numeric metrics reported in this text.")
            else:
                cols = st.columns(min(len(mets), 4))
                for i, (_, m) in enumerate(mets.iterrows()): cols[i % len(cols)].metric(m["metric"].upper() if len(m["metric"]) <= 5 else m["metric"].title(), m["value"])
        with t4:
            raw = ie["llm"]
            try:
                j = json.loads(raw[raw.index("{"):raw.rindex("}") + 1]); st.markdown("**Task:** " + str(j.get("task", "–")))
                for key, label in [("methods", "Methods"), ("datasets", "Datasets"), ("metrics", "Metrics")]:
                    v = j.get(key) or ["–"]; st.markdown(f"**{label}**"); st.markdown(chips(v if isinstance(v, list) else [v]), unsafe_allow_html=True)
                st.markdown("**Key findings**")
                for f in (j.get("key_findings") or []): st.markdown(f"- {f}")
                with st.expander("Raw JSON"): st.json(j)
            except Exception:
                st.warning("Could not parse the LLM output as JSON. Showing it as returned:"); st.code(raw)
    with st.expander("Corpus-level view: sentiment by topic"):
        if st.button("Compute for 150 papers (~2 min)"):
            theta, tops, _, _ = lda_model(8); rows = []
            for i in np.random.RandomState(0).choice(len(df), 150, replace=False):
                ss = sent_split(df.abstract.iloc[i]); rr = sentiment(ss) if ss else []
                if rr: rows.append({"topic": ", ".join(tops[theta[i].argmax()][:3]), "positive": np.mean([l == "POSITIVE" for l, _ in rr])})
            g = pd.DataFrame(rows).groupby("topic").positive.mean().reset_index()
            st.plotly_chart(px.bar(g, x="positive", y="topic", orientation="h", template="scholar"), use_container_width=True, theme=None)

# ================= Research Assistant (RAG + Groq fallback, English & Hindi, text & voice) =================
import hashlib
def groq_key():
    try: return os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
    except Exception: return ""

def llm_safe(prompt):                                    # never raises: returns (text, ok)
    try:
        out = llm(prompt); return out, not out.startswith("⚠️")
    except Exception: return "", False

@st.cache_data(show_spinner=False)
def transcribe(audio, lang, key):                        # Whisper on Groq, language-hinted
    from groq import Groq
    cl = Groq(api_key=key); ids = [m.id for m in cl.models.list().data]
    model = next((m for m in ["whisper-large-v3-turbo", "whisper-large-v3"] if m in ids), next((i for i in ids if "whisper" in i), None))
    if not model: raise RuntimeError("no speech-to-text model on this Groq account")
    r = cl.audio.transcriptions.create(file=("voice.wav", audio), model=model, language=lang, response_format="text", temperature=0.0)
    return str(getattr(r, "text", r)).strip()

def translate_smart(texts, src, tgt):                    # Groq LLM first (better Hindi), MarianMT fallback
    nm = {"hi": "Hindi", "en": "English"}
    out, ok = llm_safe(f"Translate each item from {nm[src]} to {nm[tgt]} (if an item is already in {nm[tgt]}, keep it). Keep technical terms, model names and numbers "
                       f"accurate. Return ONLY a JSON array of {len(texts)} strings in the same order.\n\n" + json.dumps(list(texts), ensure_ascii=False))
    if ok:
        try:
            o = out.split("\n\n*Model")[0]; arr = json.loads(o[o.index("["):o.rindex("]") + 1])
            if len(arr) == len(texts): return [str(x) for x in arr], "Groq LLM"
        except Exception: pass
    return translate(list(texts), src, tgt), "MarianMT"

def paper_links(df_):
    for i, (_, r) in enumerate(df_.iterrows(), 1): st.markdown(f"[{i}] [{r.title}](https://arxiv.org/abs/{r.id})")

def assistant(mq, code):
    """RAG over the paper library; Groq general-knowledge fallback. Never raises."""
    eff = "hi" if len(re.findall(r"[\u0900-\u097F]", mq)) > 2 else code            # Devanagari text is handled as Hindi
    note = "Your text looks like Hindi, so it was handled as Hindi." if eff != code else None
    en = mq
    if eff == "hi":
        try: en = translate_smart([mq], "hi", "en")[0][0]                           # library is English: search in English
        except Exception: pass
    res = None
    try: res = rank(en, 5)
    except Exception: pass
    rule = ("Answer in Hindi (Devanagari script); keep technical terms, model names and [n] citations unchanged." if eff == "hi" else "Answer in English.")
    if res is not None and len(res) and res.score.max() > -1:                       # cross-encoder: something relevant exists
        good = res[res.score > -1].head(5)
        ctx = "\n".join(f"[{i}] {r.title}: {r.abstract}" for i, (_, r) in enumerate(good.iterrows(), 1))
        txt, ok = llm_safe(f"Answer using ONLY the sources below. Cite sources as [n]. {rule} If the sources do not contain the answer, reply with exactly NO_ANSWER."
                           f"\n\nSources:\n{ctx}\n\nQuestion: {mq}\n(English version: {en})")
        if ok and "NO_ANSWER" not in txt: return dict(text=txt, mode="corpus", src=good, eff=eff, en=en, note=note)
    txt, ok = llm_safe(f"You are a helpful research assistant. {rule} Be clear and concise; if unsure, say so.\n\nQuestion: {mq}")
    return dict(text=txt if ok else None, mode="groq" if ok else "none", src=res, eff=eff, en=en, note=note)

if page == NAMES[2]:
    lang = st.radio("Language / भाषा", ["English", "हिन्दी"], horizontal=True, key="ra_lang"); code = "en" if lang == "English" else "hi"
    L = lambda en_, hi_: en_ if code == "en" else hi_
    st.session_state.setdefault("ra_q", "")
    if hasattr(st, "audio_input"):
        st.markdown(L("🎙️ **Speak your question**, or type it below", "🎙️ **बोलकर पूछें**, या नीचे लिखें"))
        rec = st.audio_input("Voice", key=f"ra_mic_{code}", label_visibility="collapsed")
        if rec is not None:
            data = rec.getvalue(); tag = (hashlib.md5(data).hexdigest(), code)
            if st.session_state.get("ra_audio") != tag:
                try:
                    with st.spinner(L("Transcribing...", "आवाज़ को टेक्स्ट में बदला जा रहा है...")):
                        txt = transcribe(data, code, groq_key())
                    st.session_state.update(ra_audio=tag, ra_q=txt, ra_go=bool(txt))     # a voice question is answered automatically
                except Exception: st.warning(L("Couldn't transcribe the recording. Please type your question.", "रिकॉर्डिंग को टेक्स्ट में नहीं बदला जा सका। कृपया प्रश्न टाइप करें।"))
    else: st.caption("Voice input needs a newer Streamlit: run `pip install -U streamlit`.")
    mq = st.text_input(L("Your question", "आपका प्रश्न"), key="ra_q", placeholder=L("e.g. How do transformers handle long documents?", "जैसे: ट्रांसफ़ॉर्मर लंबे दस्तावेज़ों को कैसे संभालते हैं?")).strip()
    ex = [L("What is retrieval-augmented generation?", "रिट्रीवल-ऑगमेंटेड जनरेशन क्या है?"), L("How is image deblurring done?", "इमेज डीब्लरिंग कैसे की जाती है?"),
          L("Who invented the telephone?", "टेलीफ़ोन का आविष्कार किसने किया?")]
    for col, e in zip(st.columns(3), ex): col.button(e, key="ex_" + e, on_click=lambda e=e: st.session_state.update(ra_q=e, ra_go=True), use_container_width=True)
    if (st.button(L("Ask", "पूछें"), type="primary") or st.session_state.pop("ra_go", False)) and mq:
        adb.log_search(uid, "assistant", mq)
        with st.spinner(L("Searching the research papers...", "शोध पत्रों में खोज रहे हैं...")): st.session_state["ra_out"] = assistant(mq, code)
    out = st.session_state.get("ra_out")
    if out:
        hi = out["eff"] == "hi"; src = out["src"]
        if out["note"]: st.caption(out["note"])
        if hi and out["en"]: st.caption(f"Search query (English): {out['en']}")
        if out["mode"] == "corpus":
            st.success("📚 Based on the research library" if not hi else "📚 शोध-पत्र संग्रह पर आधारित उत्तर"); st.markdown(out["text"])
            st.markdown("**Sources / स्रोत**" if hi else "**Sources**"); paper_links(src)
            if hi: st.caption("Source titles are in English, as in the original papers.")
        elif out["mode"] == "groq":
            st.warning("🌐 General-knowledge answer (Groq). No relevant papers were found in the research library, so this is NOT based on the indexed papers."
                       if not hi else "🌐 सामान्य ज्ञान पर आधारित उत्तर (Groq)। शोध-पत्र संग्रह में कोई प्रासंगिक पेपर नहीं मिला, इसलिए यह उत्तर संग्रहित पेपरों पर आधारित नहीं है।")
            st.markdown(out["text"])
            if src is not None and len(src):
                with st.expander("Closest papers in the library (low relevance)"): paper_links(src)
        else:
            st.info("The answer service isn't reachable right now. Check your Groq key and internet connection, then try again."
                    if not hi else "उत्तर सेवा अभी उपलब्ध नहीं है। अपनी Groq key और इंटरनेट जाँचें, फिर दोबारा प्रयास करें।")
            if src is not None and len(src): st.markdown("**Closest papers in the library:**"); paper_links(src)

if page == NAMES[6]:   # ---- My Library (saved papers, history, activity, account)
    t1, t2, t3, t4 = st.tabs(["⭐ Saved papers", "🕘 Search history", "📈 Activity", "⚙️ Account"])
    with t1:
        rows = adb.list_saved(uid); st.caption(f"{len(rows)} saved paper(s)")
        if rows: st.download_button("Download as CSV", pd.DataFrame(rows).to_csv(index=False), "saved_papers.csv", "text/csv")
        else: st.info("Papers you save with ☆ Save to library will appear here.")
        for r_ in rows:
            with st.expander(r_["title"]):
                st.caption(f"arXiv:{r_['paper_id']} · {r_['categories']} · saved {r_['saved_at']}"); st.write(r_["abstract"])
                c1, c2 = st.columns(2); c1.link_button("Open on arXiv ↗", f"https://arxiv.org/abs/{r_['paper_id']}")
                c2.button("Remove", key=f"rm_{r_['paper_id']}", on_click=lambda pid=r_["paper_id"]: (adb.unsave_paper(uid, pid), adb.log_activity(uid, "unsave_paper", pid)))
    with t2:
        h = adb.history(uid)
        if h:
            st.dataframe(pd.DataFrame(h).rename(columns={"created_at": "When", "kind": "Type", "query": "Query"}), use_container_width=True, hide_index=True)
            st.button("Clear search history", on_click=lambda: (adb.clear_history(uid), adb.log_activity(uid, "clear_history")))
        else: st.info("No searches yet.")
    with t3:
        cn = adb.activity_counts(uid)
        for col, (lb, k_) in zip(st.columns(4), [("Searches", "search"), ("Assistant questions", "assistant"), ("Papers saved", "save_paper"), ("Summaries", "summarize")]): col.metric(lb, cn.get(k_, 0))
        ac = adb.activity(uid)
        if ac: st.dataframe(pd.DataFrame(ac).rename(columns={"created_at": "When", "action": "Action", "detail": "Details"}), use_container_width=True, hide_index=True)
    with t4:
        st.markdown(f"**{user['name']}**  \n{user['email']}"); st.markdown("##### Change password")
        with st.form("pw_form"):
            o = st.text_input("Current password", type="password"); n1 = st.text_input("New password", type="password"); n2 = st.text_input("Confirm new password", type="password")
            go_pw = st.form_submit_button("Update password", type="primary")
        if go_pw:
            if n1 != n2: st.error("New passwords do not match.")
            else:
                ok, msg = adb.change_password(uid, o, n1)
                (st.success if ok else st.error)(msg)
                if ok: adb.log_activity(uid, "change_password")