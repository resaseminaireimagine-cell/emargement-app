import streamlit as st
import pandas as pd
from datetime import datetime
import re
import io
from pathlib import Path

# =========================
# CONFIG / BRANDING
# =========================
APP_TITLE = "Outil d’émargement — Institut Imagine"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Couleurs (ajuste si besoin)
PRIMARY = "#C4007A"   # rose Imagine (proche du logo)
BG = "#F6F7FB"
TEXT = "#111827"
MUTED = "#6B7280"

# Logo : on essaie plusieurs noms (au cas où)
LOGO_CANDIDATES = [
    "logo_rose.png",
    "LOGO_ROSE.png",
    "LOGO ROSE.png",
    "logo.png",
    "logo-imagine.png",
]

def find_logo_path() -> str | None:
    for name in LOGO_CANDIDATES:
        p = Path(name)
        if p.exists():
            return str(p)
    return None

# CSS pour un rendu plus “pro”
st.markdown(
    f"""
<style>
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.5rem; max-width: 1250px; }}

h1, h2, h3, h4 {{ color: {TEXT}; }}
small, .stCaption, p {{ color: {MUTED}; }}

/* Boutons : texte blanc forcé */
.stButton > button {{
  background: {PRIMARY};
  color: white !important;
  border: 0;
  border-radius: 12px;
  padding: 0.55rem 0.9rem;
  font-weight: 650;
}}
.stButton > button:hover {{
  filter: brightness(0.95);
}}

.stTextInput input {{
  border-radius: 12px;
}}

[data-testid="stHorizontalBlock"] {{
  background: white;
  border-radius: 14px;
  padding: 0.35rem 0.6rem;
  margin-bottom: 0.35rem;
  box-shadow: 0 1px 10px rgba(0,0,0,0.06);
}}
</style>
""",
    unsafe_allow_html=True
)

# =========================
# HELPERS DATA
# =========================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

ALIASES = {
    "first_name": ["first_name", "firstname", "prenom", "prénom", "given name", "given_name"],
    "last_name":  ["last_name", "lastname", "nom", "surname", "family name", "family_name"],
    "email":      ["email", "e-mail", "mail", "courriel"],
    "company":    ["company", "societe", "société", "organisation", "organization", "structure"],
    "function":   ["fonction", "function", "job", "poste", "title"],
}

STANDARD_ORDER = ["first_name", "last_name", "email", "company", "function"]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    original_cols = list(df.columns)
    norm_cols = {c: normalize(c) for c in original_cols}

    mapping = {}
    used_std = set()

    for std, candidates in ALIASES.items():
        for c in original_cols:
            if norm_cols[c] in candidates and std not in used_std:
                mapping[c] = std
                used_std.add(std)
                break

    return df.rename(columns=mapping)

def ensure_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "present" not in df.columns:
        df["present"] = False
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
    comp = str(row.get("company", "")).strip().lower()
    return f"name:{ln}|{fn}|{comp}"

def build_search_text(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for c in cols:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and str(v).strip():
                parts.append(str(v))
    return " ".join(parts).lower()

def load_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    df = standardize_columns(df)
    df = ensure_internal_columns(df)
    df = df.fillna("")

    # ID unique garanti (évite StreamlitDuplicateElementKey)
    df["__base_id"] = df.apply(make_base_id, axis=1)
    df["__id"] = df["__base_id"] + "|row:" + df.index.astype(str)
    df = df.drop(columns=["__base_id"], errors="ignore")

    return df

# =========================
# HEADER (LOGO + TITRE)
# =========================
logo_path = find_logo_path()

h1, h2 = st.columns([1, 5], vertical_alignment="center")
with h1:
    if logo_path:
        # LOGO réduit (au lieu de prendre toute la colonne)
        st.image(logo_path, width=140)
with h2:
    st.markdown(f"## {APP_TITLE}")
    st.caption("Importez votre liste, recherchez un participant, émargez, puis exportez la feuille d’émargement.")

st.divider()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Réglages")
    staff_name = st.text_input("Nom de l'agent (optionnel)", placeholder="Ex: Ambroise").strip()
    st.caption("Ce nom sera enregistré dans la colonne checkin_by.")
    st.markdown("---")
    st.caption("Astuce Excel : l’export CSV utilise **;** (compatibilité Excel FR).")

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("➡️ Importez un fichier Excel pour commencer.")
    st.stop()

# Charger une seule fois par fichier (évite relecture à chaque clic)
if "df" not in st.session_state or st.session_state.get("filename") != uploaded.name:
    st.session_state.df = load_excel(uploaded)
    st.session_state.filename = uploaded.name

df = st.session_state.df

# =========================
# KPIs + CONTROLS
# =========================
total = len(df)
present_count = int(df["present"].sum())

k1, k2, k3, k4 = st.columns([1, 1, 1, 2], vertical_alignment="center")
k1.metric("Participants", total)
k2.metric("Présents", present_count)
k3.metric("Restants", total - present_count)

with k4:
    query = st.text_input("Recherche", placeholder="Nom, prénom, email, société…").strip().lower()

filters = st.columns([1, 1, 2], vertical_alignment="center")
with filters[0]:
    only_not_present = st.checkbox("Non émargés", value=True)
with filters[1]:
    show_present_only = st.checkbox("Présents uniquement", value=False)
with filters[2]:
    st.caption("Affichage optimisé : 1 ligne = 1 participant")

# =========================
# FILTER / SORT
# =========================
display_cols = [c for c in STANDARD_ORDER if c in df.columns]
if not display_cols:
    display_cols = [c for c in df.columns if c not in ["present", "checkin_time", "checkin_by", "__id"]][:4]

search_cols = list(dict.fromkeys(display_cols + [c for c in ["email", "company", "function"] if c in df.columns]))

view = df.copy()

if query:
    mask = view.apply(lambda r: query in build_search_text(r, search_cols), axis=1)
    view = view[mask].copy()

if show_present_only:
    view = view[view["present"] == True].copy()
elif only_not_present:
    view = view[view["present"] == False].copy()

# Tri stable
if "last_name" in view.columns and "first_name" in view.columns:
    view = view.sort_values(by=["last_name", "first_name"], kind="stable")
elif "last_name" in view.columns:
    view = view.sort_values(by=["last_name"], kind="stable")

MAX_ROWS = 500
if len(view) > MAX_ROWS:
    st.warning(f"{len(view)} résultats. Affichage limité à {MAX_ROWS}. Affinez la recherche.")
    view = view.head(MAX_ROWS)

st.divider()

# =========================
# LIST (1 PARTICIPANT = 1 LINE)
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

for _, row in view.iterrows():
    rid = row["__id"]

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

    is_present = bool(row["present"])
    cols[5].write("✅ Présent" if is_present else "⬜ À émarger")

    if not is_present:
        # Émarger = rose (primary), Annuler = gris (default)
        if cols[6].button("Émarger", key=f"em_{rid}", use_container_width=True, type="primary"):
            idx = df.index[df["__id"] == rid]
            if len(idx):
                i = idx[0]
                df.at[i, "present"] = True
                df.at[i, "checkin_time"] = now_str()
                df.at[i, "checkin_by"] = staff_name
                st.session_state.df = df
            st.rerun()
    else:
        if cols[6].button("Annuler", key=f"an_{rid}", use_container_width=True):
            idx = df.index[df["__id"] == rid]
            if len(idx):
                i = idx[0]
                df.at[i, "present"] = False
                df.at[i, "checkin_time"] = ""
                df.at[i, "checkin_by"] = ""
                st.session_state.df = df
            st.rerun()

st.divider()

# =========================
# LAST CHECK-INS
# =========================
st.subheader("Derniers émargés")
last_df = df[df["present"] == True].copy()
if len(last_df):
    # tri par datetime texte (format YYYY-MM-DD HH:MM:SS => tri lexical OK)
    last_df = last_df.sort_values(by=["checkin_time"], ascending=False, kind="stable").head(8)
    show_cols = [c for c in ["first_name", "last_name", "company", "checkin_time", "checkin_by"] if c in last_df.columns]
    st.dataframe(last_df.drop(columns=["__id"], errors="ignore")[show_cols], use_container_width=True, hide_index=True)
else:
    st.caption("Aucun émargement pour l’instant.")

st.divider()

# =========================
# EXPORTS (CSV ; + XLSX)
# =========================
st.subheader("Exports")

export_df = df.drop(columns=["__id"], errors="ignore").copy()

ex1, ex2, ex3 = st.columns([1, 1, 1], vertical_alignment="center")

# CSV compat Excel FR (; + BOM)
with ex1:
    csv_bytes = export_df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ CSV (Excel FR)",
        data=csv_bytes,
        file_name="emargement_export.csv",
        mime="text/csv",
        use_container_width=True
    )

# CSV présents uniquement
with ex2:
    present_only = export_df[export_df["present"] == True].copy()
    csv_p_bytes = present_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ CSV (présents)",
        data=csv_p_bytes,
        file_name="emargement_presents.csv",
        mime="text/csv",
        use_container_width=True
    )

# Excel (le plus clean)
with ex3:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Emargement")
        present_only.to_excel(writer, index=False, sheet_name="Presents")
    st.download_button(
        "⬇️ Excel (.xlsx)",
        data=buffer.getvalue(),
        file_name="emargement_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with st.expander("Aperçu : présents"):
    st.dataframe(present_only, use_container_width=True, hide_index=True)
