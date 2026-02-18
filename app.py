# app.py
import io
import re
import hashlib
import html
import unicodedata
import urllib.parse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import altair as alt


# =========================
# CONFIG
# =========================
APP_TITLE = "Outil d’émargement — Institut Imagine"
PRIMARY = "#C4007A"   # rose Imagine
BG = "#F6F7FB"
TEXT = "#111827"
MUTED = "#6B7280"

PARIS_TZ = ZoneInfo("Europe/Paris")

# Autosave (sans action utilisateur)
AUTOSAVE_DIR = Path("/tmp/imagine_emargement_autosave")
AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)

LOGO_CANDIDATES = [
    "logo_rose.png",
    "LOGO ROSE.png",
    "LOGO_ROSE.png",
    "logo.png",
]

ALIASES = {
    "first_name": [
        "first_name", "firstname", "first name", "given name", "given_name",
        "prenom", "prénom"
    ],
    "last_name": [
        "last_name", "lastname", "last name", "surname", "family name", "family_name",
        "nom"
    ],
    "email": ["email", "e-mail", "mail", "courriel"],
    "company": ["company", "societe", "société", "organisation", "organization", "structure"],
    "function": ["fonction", "function", "job", "poste", "title"],
    "present": ["present", "présent", "présence", "presence"],
    "checkin_time": ["checkin_time", "checkin time", "heure", "date", "datetime", "check-in time"],
    "checkin_by": ["checkin_by", "checkin by", "agent", "émargé par", "emarge par", "checked in by"],
}
STANDARD_ORDER = ["first_name", "last_name", "email", "company", "function"]

PRESENT_TRUE = {"true", "1", "yes", "oui", "vrai", "x", "present", "présent"}

MAIL_TO = "evenements@institutimagine.org"


# =========================
# HELPERS
# =========================
def now_paris_str() -> str:
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S")


def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00A0", " ")
    s = s.replace("\t", " ").replace("\n", " ")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def fold_text(s: str) -> str:
    """Minuscule + sans accents + alphanum/espaces normalisés (recherche robuste)."""
    s = "" if s is None else str(s)
    s = s.replace("\u00A0", " ")
    s = s.replace("\t", " ").replace("\n", " ")
    s = s.strip().lower()

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = re.sub(r"[’'`´-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def query_tokens(q: str) -> list[str]:
    qf = fold_text(q)
    return [t for t in qf.split(" ") if t]


def find_logo_path() -> str | None:
    for name in LOGO_CANDIDATES:
        p = Path(name)
        if p.exists():
            return str(p)
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    original_cols = list(df.columns)
    norm_cols = {c: norm(c) for c in original_cols}

    mapping: dict[str, str] = {}
    used_std: set[str] = set()

    for std, candidates in ALIASES.items():
        candidates_norm = [norm(x) for x in candidates]
        for c in original_cols:
            nc = norm_cols[c]
            if std in used_std:
                continue
            if (
                nc in candidates_norm
                or any(nc.startswith(cand) for cand in candidates_norm)
                or any(cand in nc for cand in candidates_norm)
            ):
                mapping[c] = std
                used_std.add(std)
                break

    return df.rename(columns=mapping)


def coerce_present(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = norm(v)
    if s == "" or s == "nan":
        return False
    return s in PRESENT_TRUE


def ensure_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "present" not in df.columns:
        df["present"] = False
    else:
        df["present"] = df["present"].apply(coerce_present)

    if "checkin_time" not in df.columns:
        df["checkin_time"] = ""
    if "checkin_by" not in df.columns:
        df["checkin_by"] = ""
    return df


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.fillna("")
    for c in ["first_name", "last_name", "email", "company", "function", "checkin_time", "checkin_by", "__base_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace("nan", "")
    return df


def make_base_id(row: pd.Series) -> str:
    email = norm(row.get("email", ""))
    if email and email != "nan":
        return f"email:{email}"
    fn = norm(row.get("first_name", ""))
    ln = norm(row.get("last_name", ""))
    co = norm(row.get("company", ""))
    return f"name:{ln}|{fn}|{co}"


def add_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "__base_id" not in df.columns:
        df["__base_id"] = df.apply(make_base_id, axis=1)
    df["__id"] = df["__base_id"] + "|row:" + df.index.astype(str)
    return df


def hash_uploaded_file(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return hashlib.sha256(data).hexdigest()[:16]


def autosave_path(file_hash: str) -> Path:
    return AUTOSAVE_DIR / f"autosave_{file_hash}.csv"


def autosave_df(df: pd.DataFrame, file_hash: str) -> None:
    p = autosave_path(file_hash)
    export_df = df.drop(columns=["__id"], errors="ignore").copy()
    export_df.to_csv(p, index=False, sep=";", encoding="utf-8-sig")


def try_load_autosave(file_hash: str) -> pd.DataFrame | None:
    p = autosave_path(file_hash)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, sep=";", encoding="utf-8-sig")
        df = standardize_columns(df)
        df = ensure_internal_columns(df)
        df = sanitize_df(df)
        df = add_ids(df)
        return df
    except Exception as err:
        st.session_state["autosave_error"] = f"Autosauvegarde ignorée ({err.__class__.__name__})"
        return None


def upsert_search_blob(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """(Re)construit le blob de recherche si nécessaire."""
    df = df.copy()
    search_blob_key = "||".join(cols)
    if "_search_blob" not in df.columns or df.attrs.get("search_blob_key") != search_blob_key:
        df["_search_blob"] = build_search_blob(df, cols)
        df.attrs["search_blob_key"] = search_blob_key
    return df


def set_presence(
    df: pd.DataFrame,
    rid: str,
    present: bool,
    staff_name: str,
) -> pd.DataFrame:
    """Applique un émargement/annulation sur une ligne identifiée par __id."""
    idx = df.index[df["__id"] == rid]
    if not len(idx):
        return df

    i = idx[0]
    df = df.copy()
    df.at[i, "present"] = present
    if present:
        df.at[i, "checkin_time"] = now_paris_str()
        df.at[i, "checkin_by"] = staff_name if staff_name else "unknown"
    else:
        df.at[i, "checkin_time"] = ""
        df.at[i, "checkin_by"] = ""
    return df


def escape_cell(value) -> str:
    return html.escape(str(value or ""))


def load_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    df = standardize_columns(df)
    df = ensure_internal_columns(df)
    df = sanitize_df(df)
    df = add_ids(df)
    return df


def build_search_blob(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = []
    for c in cols:
        if c in df.columns:
            parts.append(df[c].astype(str))
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    blob = parts[0]
    for p in parts[1:]:
        blob = blob + " " + p
    return blob.map(fold_text)


def relevance_score_row(row: pd.Series, q: str) -> int:
    toks = query_tokens(q)
    if not toks:
        return 0

    fn = fold_text(row.get("first_name", ""))
    ln = fold_text(row.get("last_name", ""))
    em = fold_text(row.get("email", ""))
    co = fold_text(row.get("company", ""))
    fu = fold_text(row.get("function", ""))

    score = 0
    full_name = (fn + " " + ln).strip()
    rev_name = (ln + " " + fn).strip()
    qf = " ".join(toks)

    if qf and (qf == full_name or qf == rev_name):
        score += 250

    for t in toks:
        if t == em:
            score += 200
        if t == ln:
            score += 120
        if t == fn:
            score += 100

        if ln.startswith(t):
            score += 90
        if fn.startswith(t):
            score += 70
        if em.startswith(t):
            score += 60
        if co.startswith(t):
            score += 40

        if t in ln:
            score += 50
        if t in fn:
            score += 35
        if t in em:
            score += 30
        if t in co:
            score += 20
        if t in fu:
            score += 10

    if not bool(row.get("present", False)):
        score += 5

    return score


def build_exports(df: pd.DataFrame) -> tuple[bytes, bytes, bytes]:
    export_df = df.drop(columns=["__id"], errors="ignore").copy()

    csv_all = export_df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    present_only = export_df[export_df["present"] == True].copy()
    csv_present = present_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Emargement")
        present_only.to_excel(writer, index=False, sheet_name="Presents")
    xlsx = buffer.getvalue()

    return csv_all, csv_present, xlsx


def pager(page_count: int, page_value: int, label: str):
    c_prev, c_info, c_next = st.columns([1, 2, 1], vertical_alignment="center")
    with c_prev:
        if st.button("Page précédente", disabled=(page_value <= 1), key=f"prev_{label}", use_container_width=True):
            st.session_state.page = max(1, page_value - 1)
            st.rerun()
    with c_info:
        st.markdown(
            f"<div style='text-align:center; font-weight:800; padding:0.35rem 0;'>Page {page_value} / {page_count}</div>",
            unsafe_allow_html=True,
        )
    with c_next:
        if st.button("Page suivante", disabled=(page_value >= page_count), key=f"next_{label}", use_container_width=True):
            st.session_state.page = min(page_count, page_value + 1)
            st.rerun()


def badge_html(is_present: bool) -> str:
    if is_present:
        return "<span class='badge-present'>✔ Présent</span>"
    return "<span class='badge-todo'>À émarger</span>"


def mailto_link(to: str, subject: str, body: str) -> str:
    params = {"subject": subject, "body": body}
    return f"mailto:{to}?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


# =========================
# PAGE CONFIG + CSS
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Force le chargement des icônes (peut aider selon versions / cache)
st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
""", unsafe_allow_html=True)

css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&display=swap');

:root {{
  --font: 'Montserrat', sans-serif;
}}

html, body, .stApp, [class*="css"] {{
  font-family: var(--font) !important;
}}

h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown *,
.stCaption, small, p, label, span, div,
button, input, textarea {{
  font-family: var(--font) !important;
}}

/* Cache le header natif Streamlit (évite le doublon) */
header[data-testid="stHeader"] {{
  display: none;
}}

.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.1rem; max-width: 1280px; }}
h1, h2, h3, h4 {{ color: {TEXT}; }}
.stCaption, small, p {{ color: {MUTED}; }}

[data-testid="stHorizontalBlock"] {{
  background: white;
  border-radius: 16px;
  padding: 0.50rem 0.75rem;
  margin-bottom: 0.50rem;
  box-shadow: 0 1px 12px rgba(0,0,0,0.06);
}}

.stTextInput input {{
  border-radius: 14px;
  padding: 0.7rem 0.9rem;
  font-size: 1.0rem;
}}

.stButton > button {{
  background-color: {PRIMARY} !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 0.85rem 1.05rem !important;
  font-weight: 900 !important;
  min-height: 52px !important; /* tablette */
  white-space: nowrap !important;
}}
.stButton > button * {{ color: #ffffff !important; }}

button[kind="secondary"], .stButton > button[kind="secondary"] {{
  background: #ffffff !important;
  color: {PRIMARY} !important;
  border: 2px solid {PRIMARY} !important;
  white-space: nowrap !important;
}}
button[kind="secondary"] * {{ color: {PRIMARY} !important; }}

.badge-present {{
  background:#DCFCE7; color:#166534; padding:7px 12px; border-radius:10px; font-weight:900;
  display:inline-block; white-space:nowrap;
}}
.badge-todo {{
  background:#F3F4F6; color:#374151; padding:7px 12px; border-radius:10px; font-weight:900;
  display:inline-block; white-space:nowrap;
}}

.cell-nowrap {{
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}}

/* Fix : évite l'affichage "double_arrow_right" en texte */
button[data-testid="stSidebarCollapseButton"] span {{
  font-size: 0 !important;
}}

/* Fallback propre (offline) si les icônes ne chargent pas */
button[data-testid="stSidebarCollapseButton"]::before {{
  content: "❯❯";
  font-weight: 900;
  font-size: 18px;
  color: {MUTED};
}}

@media (max-width: 980px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  .stTextInput input {{ font-size: 1.07rem !important; }}
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# =========================
# HEADER (custom premium)
# =========================
logo_path = find_logo_path()
c1, c2 = st.columns([1, 6], vertical_alignment="center")
with c1:
    if logo_path:
        st.image(logo_path, width=90)
with c2:
    st.markdown(f"## {APP_TITLE}")
    st.caption("Importez votre liste, recherchez un participant, émargez, puis exportez la feuille d’émargement.")
st.divider()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Réglages")
    staff_name = st.text_input("Nom de l'agent", placeholder="Ex: Doralis").strip()
    st.caption("Sera enregistré dans la colonne checkin by")
    st.markdown("---")
    tablet_mode = st.toggle("Mode tablette (touch)", value=True)
    st.markdown("---")
    st.caption("Fuseau horaire : **Europe/Paris**")
    st.caption("Autosauvegarde automatique")

# =========================
# UPLOAD + AUTO-RESTORE
# =========================
uploaded = st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Importez un fichier Excel pour commencer.")
    st.stop()

file_hash = hash_uploaded_file(uploaded)

if "file_hash" not in st.session_state or st.session_state.get("file_hash") != file_hash:
    st.session_state.file_hash = file_hash
    st.session_state.filename = uploaded.name
    st.session_state.page = 1
    st.session_state["_prev_query"] = ""
    st.session_state["_prev_page_size"] = None

    restored = try_load_autosave(file_hash)
    if restored is not None:
        st.session_state.df = restored
        st.toast("Autosauvegarde restaurée ✅", icon="✅")
    else:
        st.session_state.df = load_excel(uploaded)
        autosave_df(st.session_state.df, file_hash)

autosave_error = st.session_state.pop("autosave_error", None)
if autosave_error:
    st.warning(autosave_error)

df = st.session_state.df

# =========================
# INFO
# =========================
st.caption("💾 Autosauvegarde : l’état est enregistré automatiquement après chaque émargement/annulation.")
st.divider()

# =========================
# DASHBOARD
# =========================
total = len(df)
present_count = int(df["present"].sum())
remaining = total - present_count

st.subheader("Tableau de bord")

progress_df = pd.DataFrame({"Statut": ["Présents", "Restants"], "Nombre": [present_count, remaining]})
donut = (
    alt.Chart(progress_df)
    .mark_arc(innerRadius=70)
    .encode(
        theta=alt.Theta("Nombre:Q"),
        color=alt.Color("Statut:N", legend=alt.Legend(title=None)),
        tooltip=["Statut:N", "Nombre:Q"],
    )
    .properties(height=240)
)
st.altair_chart(donut, use_container_width=True)
st.divider()

# =========================
# KPI + SEARCH + FILTERS
# =========================
k1, k2, k3, k4 = st.columns([1, 1, 1, 2], vertical_alignment="center")
k1.metric("Participants", total)
k2.metric("Présents", present_count)
k3.metric("Restants", remaining)

with k4:
    query = st.text_input(
        "Recherche",
        placeholder="Nom, prénom, email, société… (accents ignorés, ordre libre)",
        key="search_query",
    ).strip()

prev_q = st.session_state.get("_prev_query", "")
if query != prev_q:
    st.session_state.page = 1
st.session_state["_prev_query"] = query

f1, f2, f3 = st.columns([1, 1, 2], vertical_alignment="center")
with f1:
    only_not_present = st.checkbox("Non émargés", value=True)
with f2:
    show_present_only = st.checkbox("Présents uniquement", value=False)
with f3:
    st.caption("Astuce : tape 'dupont jean' ou 'jean dupont' (accents/apostrophes ignorés).")

st.divider()

# =========================
# VIEW FILTER + SORT
# =========================
display_cols = [c for c in STANDARD_ORDER if c in df.columns]
if not display_cols:
    display_cols = [
        c for c in df.columns
        if c not in ["present", "checkin_time", "checkin_by", "__id", "__base_id", "_search_blob"]
    ][:4]

search_cols = list(dict.fromkeys(display_cols + [c for c in ["email", "company", "function"] if c in df.columns]))

# Build search blob if missing/outdated (for fast AND-token filtering)
df = upsert_search_blob(df, search_cols)
if not df.equals(st.session_state.df):
    st.session_state.df = df
    autosave_df(df, file_hash)

view = df.copy()

if query.strip():
    toks = query_tokens(query)
    mask = pd.Series(True, index=view.index)
    for t in toks:
        mask &= view["_search_blob"].str.contains(re.escape(t), na=False, regex=True)
    view = view[mask].copy()

    view["_score"] = view.apply(lambda r: relevance_score_row(r, query), axis=1)
    view = view.sort_values(by=["_score"], ascending=False, kind="stable")
else:
    if "last_name" in view.columns and "first_name" in view.columns:
        view = view.sort_values(by=["last_name", "first_name"], kind="stable")

if show_present_only:
    view = view[view["present"] == True].copy()
elif only_not_present:
    view = view[view["present"] == False].copy()

# Auto-target if single non-present result
auto_target_id = None
if query.strip():
    candidates = view[view["present"] == False]
    if len(candidates) == 1:
        auto_target_id = candidates.iloc[0]["__id"]

# =========================
# QUICK TARGET CARD (tablet friendly)
# =========================
if auto_target_id:
    target_row = df[df["__id"] == auto_target_id].iloc[0]
    st.markdown("### 🎯 Participant trouvé")
    cA, cB = st.columns([5, 2], vertical_alignment="center")

    with cA:
        st.markdown(
            f"""
            <div style="background:white;border-radius:16px;padding:14px 16px;
                        box-shadow:0 1px 12px rgba(0,0,0,0.06);">
              <div style="font-weight:900;font-size:1.15rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                {target_row.get("first_name","")} {target_row.get("last_name","")}
              </div>
              <div style="color:{MUTED};margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                {target_row.get("email","")} • {target_row.get("company","")} • {target_row.get("function","")}
              </div>
              <div style="margin-top:10px;">
                <span class='badge-todo'>À émarger</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cB:
        if st.button("✅ Émarger maintenant", key=f"quick_em_{auto_target_id}", use_container_width=True, type="primary"):
            df = set_presence(df, auto_target_id, present=True, staff_name=staff_name)
            st.session_state.df = df
            autosave_df(df, file_hash)
            st.rerun()

    st.divider()

# =========================
# PAGINATION
# =========================
PAGE_SIZE_OPTIONS = [25, 50, 75, 100]

default_page_size = 25 if tablet_mode else 50
if default_page_size not in PAGE_SIZE_OPTIONS:
    default_page_size = PAGE_SIZE_OPTIONS[0]

PAGE_SIZE = st.selectbox(
    "Participants par page",
    PAGE_SIZE_OPTIONS,
    index=PAGE_SIZE_OPTIONS.index(default_page_size),
    key="page_size",
)

prev_ps = st.session_state.get("_prev_page_size", PAGE_SIZE)
if PAGE_SIZE != prev_ps:
    st.session_state.page = 1
st.session_state["_prev_page_size"] = PAGE_SIZE

total_rows = len(view)
page_count = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

if "page" not in st.session_state:
    st.session_state.page = 1
st.session_state.page = min(max(1, st.session_state.page), page_count)

pager(page_count, st.session_state.page, label="top")

start = (st.session_state.page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
view_page = view.iloc[start:end].copy()

# =========================
# LIST
# =========================
st.subheader("Liste des participants")

header = st.columns([2.2, 2.8, 3, 3, 3, 2, 2], vertical_alignment="center")
header[0].markdown("**Prénom**")
header[1].markdown("**Nom**")
header[2].markdown("**Email**")
header[3].markdown("**Société**")
header[4].markdown("**Fonction**")
header[5].markdown("**Statut**")
header[6].markdown("**Action**")

for _, row in view_page.iterrows():
    rid = row["__id"]
    is_present = bool(row["present"])

    fn = row.get("first_name", "")
    ln = row.get("last_name", "")
    em = row.get("email", "")
    co = row.get("company", "")
    fu = row.get("function", "")

    fn_safe = escape_cell(fn)
    ln_safe = escape_cell(ln)
    em_safe = escape_cell(em)
    co_safe = escape_cell(co)
    fu_safe = escape_cell(fu)

    cols = st.columns([2.2, 2.8, 3, 3, 3, 2, 2], vertical_alignment="center")

    cols[0].markdown(f"<div class='cell-nowrap'>{fn_safe}</div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='cell-nowrap'>{ln_safe}</div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='cell-nowrap'>{em_safe}</div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='cell-nowrap'>{co_safe}</div>", unsafe_allow_html=True)
    cols[4].markdown(f"<div class='cell-nowrap'>{fu_safe}</div>", unsafe_allow_html=True)

    cols[5].markdown(badge_html(is_present), unsafe_allow_html=True)

    if not is_present:
        if cols[6].button("Émarger", key=f"em_{rid}", use_container_width=True, type="primary"):
            df = set_presence(df, rid, present=True, staff_name=staff_name)
            st.session_state.df = df
            autosave_df(df, file_hash)
            st.rerun()
    else:
        if cols[6].button("Annuler", key=f"an_{rid}", use_container_width=True, type="secondary"):
            df = set_presence(df, rid, present=False, staff_name=staff_name)
            st.session_state.df = df
            autosave_df(df, file_hash)
            st.rerun()

pager(page_count, st.session_state.page, label="bottom")

if total_rows == 0:
    st.caption("Affichage : 0 / 0")
else:
    st.caption(f"Affichage : {start+1}-{min(end, total_rows)} / {total_rows}")

st.divider()

# =========================
# EXPORTS
# =========================
st.subheader("Exports")

csv_all, csv_present, xlsx_all = build_exports(df)

e1, e2, e3 = st.columns([1, 1, 1], vertical_alignment="center")
with e1:
    st.download_button(
        "⬇️ CSV (Excel FR)",
        data=csv_all,
        file_name="emargement_export.csv",
        mime="text/csv",
        use_container_width=True,
    )
with e2:
    st.download_button(
        "⬇️ CSV (présents)",
        data=csv_present,
        file_name="emargement_presents.csv",
        mime="text/csv",
        use_container_width=True,
    )
with e3:
    st.download_button(
        "⬇️ Excel (.xlsx)",
        data=xlsx_all,
        file_name="emargement_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.divider()

# =========================
# EMAIL (ouvre l'app mail par défaut)
# =========================
st.subheader("Envoyer les exports par email")

export_df = df.drop(columns=["__id"], errors="ignore").copy()
present_only = export_df[export_df["present"] == True].copy()
not_present_only = export_df[export_df["present"] == False].copy()

csv_present_bytes = present_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
csv_not_present_bytes = not_present_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")

# Tablet-friendly: empilement vertical pour éviter chevauchement
st.download_button(
    "⬇️ CSV (présents) pour email",
    data=csv_present_bytes,
    file_name="emargement_presents.csv",
    mime="text/csv",
    use_container_width=True,
)
st.download_button(
    "⬇️ CSV (non émargés) pour email",
    data=csv_not_present_bytes,
    file_name="emargement_non_emarges.csv",
    mime="text/csv",
    use_container_width=True,
)
st.caption("Télécharge les 2 CSV ci-dessus, puis ouvre l’email pré-rempli et joins les fichiers.")

ts = datetime.now(PARIS_TZ).strftime("%Y-%m-%d_%H%M")
subject = f"[Émargement] Exports présents / non émargés — {st.session_state.get('filename','liste')} — {ts}"
body = (
    "Bonjour,\n\n"
    "Veuillez trouver en pièces jointes :\n"
    "- la liste des présents\n"
    "- la liste des non émargés\n\n"
    "Pièces jointes à ajouter :\n"
    "1) emargement_presents.csv\n"
    "2) emargement_non_emarges.csv\n\n"
    f"Agent : {staff_name or 'unknown'}\n"
    f"Fichier : {st.session_state.get('filename','')}\n"
    f"Horodatage : {now_paris_str()} (Europe/Paris)\n"
)

link = mailto_link(MAIL_TO, subject, body)

st.markdown(
    f"""
    <a href="{link}">
      <button style="
        background:{PRIMARY};
        color:white;
        border:none;
        border-radius:14px;
        padding:0.85rem 1.05rem;
        font-weight:900;
        min-height:52px;
        width:100%;
        cursor:pointer;">
        📧 Ouvrir l’application mail
      </button>
    </a>
    """,
    unsafe_allow_html=True
)
