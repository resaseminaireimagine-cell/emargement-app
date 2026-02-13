import io
import re
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
    # colonnes internes possibles si un fichier est déjà "enrichi"
    "present": ["present", "présent", "présence", "presence"],
    "checkin_time": ["checkin_time", "checkin time", "heure", "date", "datetime", "check-in time"],
    "checkin_by": ["checkin_by", "checkin by", "agent", "émargé par", "emarge par", "checked in by"],
}
STANDARD_ORDER = ["first_name", "last_name", "email", "company", "function"]


# =========================
# HELPERS
# =========================
def now_paris_str() -> str:
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S")


def norm(s: str) -> str:
    s = str(s)
    s = s.replace("\u00A0", " ")  # espace insécable
    s = s.replace("\t", " ")
    s = s.replace("\n", " ")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def find_logo_path() -> str | None:
    for name in LOGO_CANDIDATES:
        p = Path(name)
        if p.exists():
            return str(p)
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme les colonnes selon ALIASES.
    Ex: "First name" -> first_name ; "Last name" -> last_name ; etc.
    """
    df = df.copy()
    original_cols = list(df.columns)
    norm_cols = {c: norm(c) for c in original_cols}

    mapping = {}
    used_std = set()

    for std, candidates in ALIASES.items():
        candidates_norm = set(norm(x) for x in candidates)
        for c in original_cols:
            nc = norm_cols[c]
            if (
                nc in candidates_norm
                or any(nc.startswith(cand) for cand in candidates_norm)
                or any(cand in nc for cand in candidates_norm)
            ) and std not in used_std:
                mapping[c] = std
                used_std.add(std)
                break

    return df.rename(columns=mapping)


def ensure_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "present" not in df.columns:
        df["present"] = False
    else:
        # Normaliser en bool si l'excel contient "Oui/Non", "TRUE/FALSE", etc.
        df["present"] = df["present"].apply(lambda x: str(x).strip().lower() in ["true", "1", "yes", "oui", "vrai"])
    if "checkin_time" not in df.columns:
        df["checkin_time"] = ""
    if "checkin_by" not in df.columns:
        df["checkin_by"] = ""
    return df


def make_base_id(row: pd.Series) -> str:
    email = str(row.get("email", "")).strip().lower()
    if email and email != "nan":
        return f"email:{email}"
    fn = str(row.get("first_name", "")).strip().lower()
    ln = str(row.get("last_name", "")).strip().lower()
    co = str(row.get("company", "")).strip().lower()
    return f"name:{ln}|{fn}|{co}"


def load_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    df = standardize_columns(df)
    df = ensure_internal_columns(df)
    df = df.fillna("")

    # ID unique par ligne -> évite StreamlitDuplicateElementKey
    df["__base_id"] = df.apply(make_base_id, axis=1)
    df["__id"] = df["__base_id"] + "|row:" + df.index.astype(str)
    df = df.drop(columns=["__base_id"], errors="ignore")
    return df


def search_text(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for c in cols:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and str(v).strip():
                parts.append(str(v))
    return " ".join(parts).lower()


def build_exports(df: pd.DataFrame) -> tuple[bytes, bytes, bytes]:
    export_df = df.drop(columns=["__id"], errors="ignore").copy()

    # CSV Excel FR : séparateur ;
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
        prev_disabled = page_value <= 1
        if st.button("⬅️ Page précédente", disabled=prev_disabled, key=f"prev_{label}", use_container_width=True):
            st.session_state.page = max(1, page_value - 1)
            st.rerun()
    with c_info:
        st.markdown(
            f"<div style='text-align:center; font-weight:800; padding:0.35rem 0;'>Page {page_value} / {page_count}</div>",
            unsafe_allow_html=True,
        )
    with c_next:
        next_disabled = page_value >= page_count
        if st.button("Page suivante ➡️", disabled=next_disabled, key=f"next_{label}", use_container_width=True):
            st.session_state.page = min(page_count, page_value + 1)
            st.rerun()


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")

# CSS global
css = f"""
<style>
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
  padding: 0.75rem 1.05rem !important;
  font-weight: 800 !important;
  min-height: 46px !important;
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
  background:#DCFCE7; color:#166534; padding:6px 12px; border-radius:10px; font-weight:800;
  display:inline-block; white-space:nowrap;
}}
.badge-todo {{
  background:#F3F4F6; color:#374151; padding:6px 12px; border-radius:10px; font-weight:800;
  display:inline-block; white-space:nowrap;
}}

@media (max-width: 980px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}

  /* Mode tablette : on réduit un peu la typo des badges et des boutons
     pour éviter "À émarger" / "Émarger" coupés sur 2 lignes */
  .badge-present, .badge-todo {{
    font-size: 0.90rem;
    padding: 6px 10px;
  }}
  .stButton > button {{
    min-height: 52px !important;
    font-size: 0.95rem !important;
    padding: 0.70rem 0.90rem !important;
  }}
  .stTextInput input {{ font-size: 1.05rem !important; }}
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
    st.caption("Importez votre liste, recherchez un participant, émargez, puis exportez la feuille d’émargement.")
st.divider()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Réglages")
    staff_name = st.text_input("Nom de l'agent (optionnel)", placeholder="Ex: Ambroise").strip()
    st.caption("Sera enregistré dans la colonne checkin_by.")
    st.markdown("---")
    tablet_mode = st.toggle("Mode tablette (touch)", value=True)
    st.markdown("---")
    st.caption("Fuseau horaire utilisé : **Europe/Paris** ✅")

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("➡️ Importez un fichier Excel pour commencer.")
    st.stop()

if "df" not in st.session_state or st.session_state.get("filename") != uploaded.name:
    st.session_state.df = load_excel(uploaded)
    st.session_state.filename = uploaded.name
    st.session_state.page = 1  # reset pagination à chaque nouveau fichier

df = st.session_state.df

# =========================
# DASHBOARD
# =========================
total = len(df)
present_count = int(df["present"].sum())
remaining = total - present_count

st.subheader("Tableau de bord")

# Camembert au-dessus des KPIs
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
# KPIs + SEARCH + FILTERS
# =========================
k1, k2, k3, k4 = st.columns([1, 1, 1, 2], vertical_alignment="center")
k1.metric("Participants", total)
k2.metric("Présents", present_count)
k3.metric("Restants", remaining)
with k4:
    query = st.text_input("Recherche", placeholder="Nom, prénom, email, société…").strip().lower()

f1, f2, f3 = st.columns([1, 1, 2], vertical_alignment="center")
with f1:
    only_not_present = st.checkbox("Non émargés", value=True)
with f2:
    show_present_only = st.checkbox("Présents uniquement", value=False)
with f3:
    st.caption("Affichage optimisé : 1 ligne = 1 participant")

st.divider()

# =========================
# LIST / FILTER / PAGINATION
# =========================
display_cols = [c for c in STANDARD_ORDER if c in df.columns]
if not display_cols:
    display_cols = [c for c in df.columns if c not in ["present", "checkin_time", "checkin_by", "__id"]][:4]

search_cols = list(dict.fromkeys(display_cols + [c for c in ["email", "company", "function"] if c in df.columns]))

view = df.copy()

if query:
    mask = view.apply(lambda r: query in search_text(r, search_cols), axis=1)
    view = view[mask].copy()

if show_present_only:
    view = view[view["present"] == True].copy()
elif only_not_present:
    view = view[view["present"] == False].copy()

if "last_name" in view.columns and "first_name" in view.columns:
    view = view.sort_values(by=["last_name", "first_name"], kind="stable")

PAGE_SIZE = 25 if tablet_mode else 50
total_rows = len(view)
page_count = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

if "page" not in st.session_state:
    st.session_state.page = 1
st.session_state.page = min(max(1, st.session_state.page), page_count)

# Pagination HAUT (avant la liste)
pager(page_count, st.session_state.page, label="top")

start = (st.session_state.page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
view_page = view.iloc[start:end].copy()

# =========================
# LIST
# =========================
st.subheader("Liste des participants")

header = st.columns([2, 2, 3, 3, 3, 2, 2])
header[0].markdown("**Prénom**")
header[1].markdown("**Nom**")
header[2].markdown("**Email**")
header[3].markdown("**Société**")
header[4].markdown("**Fonction**")
header[5].markdown("**Statut**")
header[6].markdown("**Action**")

def badge_html(is_present: bool) -> str:
    if is_present:
        return "<span class='badge-present'>✔ Présent</span>"
    return "<span class='badge-todo'>À émarger</span>"

for _, row in view_page.iterrows():
    rid = row["__id"]
    is_present = bool(row["present"])

    fn = row.get("first_name", "")
    ln = row.get("last_name", "")
    em = row.get("email", "")
    co = row.get("company", "")
    fu = row.get("function", "")

    cols = st.columns([2, 2, 3, 3, 3, 2, 2])
    cols[0].write(fn)
    cols[1].write(ln)
    cols[2].write(em)
    cols[3].write(co)
    cols[4].write(fu)
    cols[5].markdown(badge_html(is_present), unsafe_allow_html=True)

    if not is_present:
        if cols[6].button("Émarger", key=f"em_{rid}", use_container_width=True, type="primary"):
            idx = df.index[df["__id"] == rid]
            if len(idx):
                i = idx[0]
                df.at[i, "present"] = True
                df.at[i, "checkin_time"] = now_paris_str()  # ✅ heure Paris
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

# Pagination BAS (après la liste) -> évite de remonter
pager(page_count, st.session_state.page, label="bottom")

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
