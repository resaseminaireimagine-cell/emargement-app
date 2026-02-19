# app.py (version autonome : pas de Cloudbox, reprise via lien)
import io
import re
import json
import zlib
import base64
import hashlib
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
PRIMARY = "#C4007A"
BG = "#F6F7FB"
TEXT = "#111827"
MUTED = "#6B7280"
PARIS_TZ = ZoneInfo("Europe/Paris")

MAIL_TO = "evenements@institutimagine.org"

LOGO_CANDIDATES = ["logo_rose.png", "LOGO ROSE.png", "LOGO_ROSE.png", "logo.png"]

ALIASES = {
    "first_name": ["first_name", "firstname", "first name", "given name", "given_name", "prenom", "prénom"],
    "last_name": ["last_name", "lastname", "last name", "surname", "family name", "family_name", "nom"],
    "email": ["email", "e-mail", "mail", "courriel"],
    "company": ["company", "societe", "société", "organisation", "organization", "structure"],
    "function": ["fonction", "function", "job", "poste", "title"],
    "present": ["present", "présent", "présence", "presence"],
    "checkin_time": ["checkin_time", "checkin time", "heure", "date", "datetime", "check-in time"],
    "checkin_by": ["checkin_by", "checkin by", "agent", "émargé par", "emarge par", "checked in by"],
}
STANDARD_ORDER = ["first_name", "last_name", "email", "company", "function"]
PRESENT_TRUE = {"true", "1", "yes", "oui", "vrai", "x", "present", "présent"}

INTERNAL_COLS = {"__id", "__base_id", "_search_blob", "_score"}


# =========================
# HELPERS
# =========================
def now_paris_str() -> str:
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S")


def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00A0", " ").replace("\t", " ").replace("\n", " ")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def fold_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00A0", " ").replace("\t", " ").replace("\n", " ")
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
    df = df.copy().fillna("")
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


def drop_internal(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(INTERNAL_COLS), errors="ignore")


def hash_uploaded_file(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return hashlib.sha256(data).hexdigest()[:16]


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
    export_df = drop_internal(df).copy()

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
    return "<span class='badge-present'>✔ Présent</span>" if is_present else "<span class='badge-todo'>À émarger</span>"


def mailto_link(to: str, subject: str, body: str) -> str:
    params = {"subject": subject, "body": body}
    return f"mailto:{to}?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


# =========================
# STATE PACKING (reprise via URL)
# =========================
def state_pack(state: dict) -> str:
    raw = json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    comp = zlib.compress(raw, level=9)
    return base64.urlsafe_b64encode(comp).decode("ascii").rstrip("=")


def state_unpack(token: str) -> dict | None:
    try:
        pad = "=" * (-len(token) % 4)
        comp = base64.urlsafe_b64decode((token + pad).encode("ascii"))
        raw = zlib.decompress(comp)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def snapshot_from_df(df: pd.DataFrame) -> dict:
    # on ne stocke que ce qui change (présence + horodatage + agent) par base_id
    snap = {}
    for _, r in df.iterrows():
        bid = r.get("__base_id", "")
        if not bid:
            continue
        snap[bid] = {
            "p": bool(r.get("present", False)),
            "t": str(r.get("checkin_time", "")),
            "b": str(r.get("checkin_by", "")),
        }
    return snap


def apply_snapshot(df: pd.DataFrame, snap: dict) -> pd.DataFrame:
    df = df.copy()
    if "__base_id" not in df.columns:
        return df
    for i, r in df.iterrows():
        bid = r.get("__base_id", "")
        if bid in snap:
            df.at[i, "present"] = bool(snap[bid].get("p", False))
            df.at[i, "checkin_time"] = str(snap[bid].get("t", ""))
            df.at[i, "checkin_by"] = str(snap[bid].get("b", ""))
    return df


# =========================
# PAGE CONFIG + CSS
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&display=swap">
""", unsafe_allow_html=True)

css = f"""
<style>
:root {{ --font: 'Montserrat', sans-serif; }}
html, body, .stApp, [class*="css"] {{ font-family: var(--font) !important; }}
h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown *, .stCaption, small, p, label, span, div,
button, input, textarea {{ font-family: var(--font) !important; }}

header[data-testid="stHeader"] {{ display: none; }}

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
  min-height: 52px !important;
  white-space: nowrap !important;
}}
.stButton > button * {{ color: #ffffff !important; }}

button[kind="secondary"], .stButton > button[kind="secondary"] {{
  background: #ffffff !important;
  color: {PRIMARY} !important;
  border: 2px solid {PRIMARY} !important;
}}

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
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# =========================
# HEADER
# =========================
logo_path = find_logo_path()
c1, c2 = st.columns([1, 6], vertical_alignment="center")
with c1:
    if logo_path:
        st.image(logo_path, width=90)
with c2:
    st.markdown(f"## {APP_TITLE}")
    st.caption("Importer • Rechercher • Émarger • Exporter • (Reprise via lien)")
st.divider()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Démarrage")
    staff_name = st.text_input("Nom de l’agent", placeholder="Ex: Léa").strip()
    event_code = st.text_input("Code évènement", placeholder="Ex: JUBILE_2026-03-12").strip()
    tablet_mode = st.toggle("Mode tablette (touch)", value=True)

    st.markdown("---")
    st.subheader("Reprise")
    st.caption("Optionnel : collez un lien de reprise (token) pour récupérer l’état.")
    resume_token = st.text_area("Token de reprise", height=68, placeholder="Collez ici si besoin").strip()
    if st.button("Charger le token", use_container_width=True):
        st.session_state["_resume_token"] = resume_token
        st.toast("Token chargé ✅", icon="✅")
        st.rerun()

if not staff_name:
    st.info("➡️ Saisissez le **Nom de l’agent** pour commencer.")
    st.stop()

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

    df = load_excel(uploaded)

    # Applique token de reprise si présent
    token = st.session_state.get("_resume_token", "") or st.query_params.get("r", "")
    if token:
        payload = state_unpack(token)
        if payload and payload.get("h") == file_hash:
            df = apply_snapshot(df, payload.get("s", {}))
            st.toast("Reprise appliquée ✅", icon="✅")
        else:
            st.toast("Token invalide / mauvais fichier", icon="⚠️")

    st.session_state.df = df

df = st.session_state.df

# Token de reprise actuel (basé sur l’état)
snap = snapshot_from_df(df)
packed = state_pack({"h": file_hash, "s": snap})
st.query_params["r"] = packed

st.caption("🔁 Reprise : le lien de la page contient l’état. (Copiez l’URL pour reprendre ailleurs.)")
st.divider()

# Dashboard
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

# Search + filters
k1, k2, k3, k4 = st.columns([1, 1, 1, 2], vertical_alignment="center")
k1.metric("Participants", total)
k2.metric("Présents", present_count)
k3.metric("Restants", remaining)
with k4:
    query = st.text_input("Recherche", placeholder="Nom, prénom, email, société…", key="search_query").strip()

prev_q = st.session_state.get("_prev_query", "")
if query != prev_q:
    st.session_state.page = 1
st.session_state["_prev_query"] = query

filter_choice = st.radio("Filtre", ["Non émargés", "Tous", "Présents uniquement"], index=0, horizontal=True)
st.divider()

# View
display_cols = [c for c in STANDARD_ORDER if c in df.columns]
if not display_cols:
    display_cols = [c for c in df.columns if c not in ["present", "checkin_time", "checkin_by", "__id", "__base_id", "_search_blob"]][:4]
search_cols = list(dict.fromkeys(display_cols + [c for c in ["email", "company", "function"] if c in df.columns]))

if "_search_blob" not in df.columns:
    df["_search_blob"] = build_search_blob(df, search_cols)
    st.session_state.df = df

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

if filter_choice == "Présents uniquement":
    view = view[view["present"] == True].copy()
elif filter_choice == "Non émargés":
    view = view[view["present"] == False].copy()

# Pagination
PAGE_SIZE_OPTIONS = [25, 50, 75, 100]
default_page_size = 25 if tablet_mode else 50
PAGE_SIZE = st.selectbox("Participants par page", PAGE_SIZE_OPTIONS, index=PAGE_SIZE_OPTIONS.index(default_page_size), key="page_size")

total_rows = len(view)
page_count = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
st.session_state.page = min(max(1, st.session_state.get("page", 1)), page_count)

pager(page_count, st.session_state.page, label="top")
start = (st.session_state.page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
view_page = view.iloc[start:end].copy()

# List
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

    cols = st.columns([2.2, 2.8, 3, 3, 3, 2, 2], vertical_alignment="center")
    cols[0].markdown(f"<div class='cell-nowrap'>{fn}</div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='cell-nowrap'>{ln}</div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='cell-nowrap'>{em}</div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='cell-nowrap'>{co}</div>", unsafe_allow_html=True)
    cols[4].markdown(f"<div class='cell-nowrap'>{fu}</div>", unsafe_allow_html=True)
    cols[5].markdown(badge_html(is_present), unsafe_allow_html=True)

    if not is_present:
        if cols[6].button("Émarger", key=f"em_{rid}", use_container_width=True, type="primary"):
            idx = df.index[df["__id"] == rid]
            if len(idx):
                i = idx[0]
                df.at[i, "present"] = True
                df.at[i, "checkin_time"] = now_paris_str()
                df.at[i, "checkin_by"] = staff_name
                st.session_state.df = df
            st.rerun()
    else:
        if cols[6].button("Annuler", key=f"an_{rid}", use_container_width=True, type="secondary"):
            idx = df.index[df["__id"] == rid]
            if len(idx):
                i = idx[0]
                df.at[i, "present"] = False
                df.at[i, "checkin_time"] = ""
                df.at[i, "checkin_by"] = ""
                st.session_state.df = df
            st.rerun()

pager(page_count, st.session_state.page, label="bottom")
st.caption("Affichage : 0 / 0" if total_rows == 0 else f"Affichage : {start+1}-{min(end, total_rows)} / {total_rows}")
st.divider()

# Exports
st.subheader("Exports")
csv_all, csv_present, xlsx_all = build_exports(df)
e1, e2, e3 = st.columns([1, 1, 1], vertical_alignment="center")
with e1:
    st.download_button("⬇️ CSV (Excel FR)", data=csv_all, file_name="emargement_export.csv", mime="text/csv", use_container_width=True)
with e2:
    st.download_button("⬇️ CSV (présents)", data=csv_present, file_name="emargement_presents.csv", mime="text/csv", use_container_width=True)
with e3:
    st.download_button("⬇️ Excel (.xlsx)", data=xlsx_all, file_name="emargement_export.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

st.divider()

# Email helper (mailto)
st.subheader("Envoyer les exports par email")
export_df = drop_internal(df).copy()
present_only = export_df[export_df["present"] == True].copy()
not_present_only = export_df[export_df["present"] == False].copy()

csv_present_bytes = present_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
csv_not_present_bytes = not_present_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")

st.download_button("⬇️ CSV (présents) pour email", data=csv_present_bytes, file_name="emargement_presents.csv", mime="text/csv", use_container_width=True)
st.download_button("⬇️ CSV (non émargés) pour email", data=csv_not_present_bytes, file_name="emargement_non_emarges.csv", mime="text/csv", use_container_width=True)

ts = datetime.now(PARIS_TZ).strftime("%Y-%m-%d_%H%M")
subject = f"[Émargement] {event_code or Path(uploaded.name).stem} — présents / non émargés — {ts}"
body = (
    "Bonjour,\n\n"
    "Veuillez trouver en pièces jointes :\n"
    "- la liste des présents\n"
    "- la liste des non émargés\n\n"
    "Pièces jointes à ajouter :\n"
    "1) emargement_presents.csv\n"
    "2) emargement_non_emarges.csv\n\n"
    f"Agent : {staff_name}\n"
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
