import streamlit as st
import pandas as pd
from datetime import datetime
import re

st.set_page_config(page_title="Émargement digital", layout="wide")
st.title("Émargement digital — V2")

# -------------------------
# Helpers
# -------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# Mappe automatiquement des colonnes communes vers des noms standards
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
    used = set()

    for std, candidates in ALIASES.items():
        for c in original_cols:
            if norm_cols[c] in candidates and std not in used:
                mapping[c] = std
                used.add(std)
                break

    df = df.rename(columns=mapping)
    return df

def ensure_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "present" not in df.columns:
        df["present"] = False
    if "checkin_time" not in df.columns:
        df["checkin_time"] = ""
    if "checkin_by" not in df.columns:
        df["checkin_by"] = ""
    return df

def make_id(row: pd.Series) -> str:
    email = str(row.get("email", "")).strip().lower()
    if email and email != "nan":
        return f"email:{email}"
    fn = str(row.get("first_name", "")).strip().lower()
    ln = str(row.get("last_name", "")).strip().lower()
    comp = str(row.get("company", "")).strip().lower()
    return f"name:{ln}|{fn}|{comp}"

def compute_search_text(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for c in cols:
        if c in row.index:
            v = row[c]
            if pd.notna(v):
                parts.append(str(v))
    return " ".join(parts).lower()

def load_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    df = standardize_columns(df)
    df = ensure_internal_columns(df)
    # id stable
    df["__id"] = df.apply(make_id, axis=1)
    # évite NaN partout
    df = df.fillna("")
    return df

# -------------------------
# UI Controls
# -------------------------
with st.sidebar:
    st.header("Réglages")
    staff_name = st.text_input("Nom de l'agent (optionnel)", placeholder="Ex: Ambroise").strip()
    st.caption("Tip: renseigne ce champ une fois, il sera enregistré dans les émargements.")

uploaded = st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("➡️ Importe un fichier Excel pour démarrer.")
    st.stop()

# Charger une seule fois par fichier (évite de recharger à chaque clic)
if "df" not in st.session_state or st.session_state.get("filename") != uploaded.name:
    st.session_state.df = load_excel(uploaded)
    st.session_state.filename = uploaded.name

df = st.session_state.df

# Colonnes d'affichage (si manquantes, on filtre)
display_cols = [c for c in STANDARD_ORDER if c in df.columns]
# fallback si Excel bizarre
if not display_cols:
    display_cols = [c for c in df.columns if c not in ["present", "checkin_time", "checkin_by", "__id"]][:4]

search_cols = display_cols + [c for c in ["email", "company", "function"] if c in df.columns]

# KPIs
c1, c2, c3 = st.columns(3)
total = len(df)
present_count = int(df["present"].sum())
c1.metric("Participants", total)
c2.metric("Présents", present_count)
c3.metric("Restants", total - present_count)
st.divider()

# Recherche + filtres
left, right = st.columns([3, 1])
with left:
    query = st.text_input("Recherche", placeholder="Nom, prénom, email, société…").strip().lower()
with right:
    only_not_present = st.checkbox("Non émargés", value=False)

# Filtrer
if query:
    mask = df.apply(lambda r: query in compute_search_text(r, search_cols), axis=1)
    view = df[mask].copy()
else:
    view = df.copy()

if only_not_present:
    view = view[view["present"] == False].copy()

# Limite affichage (évite lag)
MAX_ROWS = 300
if len(view) > MAX_ROWS:
    st.warning(f"{len(view)} résultats. Affichage limité à {MAX_ROWS}. Affine la recherche.")
    view = view.head(MAX_ROWS)

# -------------------------
# Table-like display: 1 participant = 1 line
# -------------------------
st.subheader("Liste des participants")

header = st.columns([2,2,3,3,3,2,2])
header[0].markdown("**Prénom**")
header[1].markdown("**Nom**")
header[2].markdown("**Email**")
header[3].markdown("**Société**")
header[4].markdown("**Fonction**")
header[5].markdown("**Statut**")
header[6].markdown("**Action**")

for _, row in view.iterrows():
    # valeurs standardisées (si absentes -> "")
    fn = row.get("first_name", "")
    ln = row.get("last_name", "")
    em = row.get("email", "")
    co = row.get("company", "")
    fu = row.get("function", "")

    cols = st.columns([2,2,3,3,3,2,2])
    cols[0].write(fn)
    cols[1].write(ln)
    cols[2].write(em)
    cols[3].write(co)
    cols[4].write(fu)

    status = "✅ Présent" if bool(row["present"]) else "⬜ À émarger"
    cols[5].write(status)

    rid = row["__id"]

    if not bool(row["present"]):
        if cols[6].button("Émarger", key=f"em_{rid}", use_container_width=True):
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

# -------------------------
# Export
# -------------------------
st.subheader("Export")

export_df = df.copy()
export_df = export_df.drop(columns=["__id"], errors="ignore")

st.download_button(
    "Télécharger le CSV d’émargement",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name="emargement_export.csv",
    mime="text/csv",
)

# Optionnel: aperçu des présents
with st.expander("Voir uniquement les présents"):
    st.dataframe(export_df[export_df["present"] == True], use_container_width=True)
