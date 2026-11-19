"""ScholarLens authentication + per-user data (SQLite). No Streamlit imports; every query is parameterised."""
import os, re, time, hmac, hashlib, secrets, sqlite3
from datetime import datetime

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scholarlens.db")
N, R, P = 2 ** 14, 8, 1                                   # scrypt cost parameters
MAX_FAILS, LOCK_SECS = 5, 300                              # lock an account for 5 min after 5 wrong passwords
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
COMMON = {"password", "password1", "password123", "12345678", "123456789", "qwerty123", "iloveyou", "admin123", "welcome1", "letmein123"}

def _now(): return datetime.now().isoformat(" ", "seconds")

def _run(sql, args=(), fetch=None):
    c = sqlite3.connect(DB, timeout=10); c.row_factory = sqlite3.Row; c.execute("PRAGMA foreign_keys=ON")
    try:
        cur = c.execute(sql, args)
        out = [dict(r) for r in cur.fetchall()] if fetch == "all" else (dict(r) if (r := cur.fetchone()) else None) if fetch == "one" else cur.rowcount
        c.commit(); return out
    finally: c.close()

def init_db():
    c = sqlite3.connect(DB, timeout=10)
    try:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL, pw_hash TEXT NOT NULL,
            failed INTEGER DEFAULT 0, locked_until REAL DEFAULT 0, created_at TEXT, last_login TEXT);
        CREATE TABLE IF NOT EXISTS saved_papers(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            paper_id TEXT NOT NULL, title TEXT, abstract TEXT, categories TEXT, saved_at TEXT, UNIQUE(user_id, paper_id));
        CREATE TABLE IF NOT EXISTS search_history(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            kind TEXT, query TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS activity(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            action TEXT, detail TEXT, created_at TEXT);
        CREATE INDEX IF NOT EXISTS i_sp ON saved_papers(user_id); CREATE INDEX IF NOT EXISTS i_sh ON search_history(user_id); CREATE INDEX IF NOT EXISTS i_ac ON activity(user_id);""")
        c.commit()
    finally: c.close()

# ---------- passwords ----------
def hash_pw(pw):
    salt = secrets.token_bytes(16)
    return f"scrypt${N}${R}${P}${salt.hex()}${hashlib.scrypt(pw.encode(), salt=salt, n=N, r=R, p=P, dklen=32).hex()}"

def verify_pw(pw, stored):
    try:
        _, n, r, p, salt, h = stored.split("$")
        calc = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc.hex(), h)
    except Exception: return False

_DUMMY = hash_pw("not-a-real-password")                    # keeps timing similar for unknown emails

def validate_email(e): return bool(EMAIL_RE.match(e)) and len(e) <= 254

def password_issues(pw, email=""):
    need = []
    if len(pw) < 8: need.append("at least 8 characters")
    if len(pw) > 128: need.append("at most 128 characters")
    if not re.search(r"[a-z]", pw): need.append("a lowercase letter")
    if not re.search(r"[A-Z]", pw): need.append("an uppercase letter")
    if not re.search(r"\d", pw): need.append("a number")
    if not re.search(r"[^A-Za-z0-9\s]", pw): need.append("a special character")
    if re.search(r"\s", pw): need.append("no spaces")
    if pw.lower() in COMMON or (email and pw.lower() == email.split("@")[0].lower()): need.append("something less guessable")
    return need

# ---------- accounts ----------
def create_user(email, name, pw):
    email, name = email.strip().lower(), name.strip()
    if not name or len(name) > 80: return False, "Please enter your name."
    if not validate_email(email): return False, "Please enter a valid email address."
    bad = password_issues(pw, email)
    if bad: return False, "Password needs: " + ", ".join(bad) + "."
    try: _run("INSERT INTO users(email,name,pw_hash,created_at) VALUES(?,?,?,?)", (email, name, hash_pw(pw), _now()))
    except sqlite3.IntegrityError: return False, "An account with this email already exists."
    return True, "Account created."

def authenticate(email, pw):
    email = email.strip().lower(); u = _run("SELECT * FROM users WHERE email=?", (email,), "one"); generic = "Invalid email or password."
    if not u: verify_pw(pw, _DUMMY); return None, generic
    if u["locked_until"] > time.time(): return None, f"Too many failed attempts. Try again in {int((u['locked_until'] - time.time()) // 60) + 1} min."
    if verify_pw(pw, u["pw_hash"]):
        _run("UPDATE users SET failed=0, locked_until=0, last_login=? WHERE id=?", (_now(), u["id"]))
        return {"id": u["id"], "email": u["email"], "name": u["name"]}, ""
    fails = u["failed"] + 1
    if fails >= MAX_FAILS: _run("UPDATE users SET failed=0, locked_until=? WHERE id=?", (time.time() + LOCK_SECS, u["id"]))
    else: _run("UPDATE users SET failed=? WHERE id=?", (fails, u["id"]))
    return None, generic

def change_password(uid, old, new):
    u = _run("SELECT * FROM users WHERE id=?", (uid,), "one")
    if not u or not verify_pw(old, u["pw_hash"]): return False, "Current password is incorrect."
    bad = password_issues(new, u["email"])
    if bad: return False, "New password needs: " + ", ".join(bad) + "."
    _run("UPDATE users SET pw_hash=? WHERE id=?", (hash_pw(new), uid)); return True, "Password updated."

# ---------- per-user data (always filtered by user_id) ----------
def save_paper(uid, p):
    return _run("INSERT OR IGNORE INTO saved_papers(user_id,paper_id,title,abstract,categories,saved_at) VALUES(?,?,?,?,?,?)",
                (uid, str(p["id"]), p["title"], p["abstract"], p.get("categories", ""), _now()))
def unsave_paper(uid, pid): return _run("DELETE FROM saved_papers WHERE user_id=? AND paper_id=?", (uid, str(pid)))
def saved_ids(uid): return {r["paper_id"] for r in _run("SELECT paper_id FROM saved_papers WHERE user_id=?", (uid,), "all")}
def list_saved(uid): return _run("SELECT paper_id,title,abstract,categories,saved_at FROM saved_papers WHERE user_id=? ORDER BY id DESC", (uid,), "all")
def log_search(uid, kind, query):
    _run("INSERT INTO search_history(user_id,kind,query,created_at) VALUES(?,?,?,?)", (uid, kind, query[:500], _now())); log_activity(uid, kind, query[:120])
def log_activity(uid, action, detail=""): _run("INSERT INTO activity(user_id,action,detail,created_at) VALUES(?,?,?,?)", (uid, action, detail[:200], _now()))
def history(uid, n=200): return _run("SELECT created_at,kind,query FROM search_history WHERE user_id=? ORDER BY id DESC LIMIT ?", (uid, n), "all")
def clear_history(uid): return _run("DELETE FROM search_history WHERE user_id=?", (uid,))
def activity(uid, n=200): return _run("SELECT created_at,action,detail FROM activity WHERE user_id=? ORDER BY id DESC LIMIT ?", (uid, n), "all")
def activity_counts(uid): return {r["action"]: r["n"] for r in _run("SELECT action, COUNT(*) n FROM activity WHERE user_id=? GROUP BY action", (uid,), "all")}