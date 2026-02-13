import streamlit as st
import pandas as pd
from datetime import datetime
import re

st.set_page_config(page_title="Émargement digital", layout="wide")
st.title("Émargement digital — V2")

# -------------------------
# Helpers
# -------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# Colonnes "standard" (dans l'app) et leurs alias possibles dans l'Excel
ALIASES = {
    "first_name": ["first_name", "firstname", "prenom", "prénom", "given name", "given_name"],
    "last_name":  ["last_name", "lastname", "nom", "surname", "family name", "family_name"],
    "email":      ["email", "e-mail", "mail", "courriel"],
    "company":    ["company", "societe", "société", "organisation", "organization", "structure"],
    "function":   ["fonction", "function", "job", "poste", "title"],
}

STANDARD_ORDER = ["first_name", "last_name", "email", "company", "function"]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme automatiquement les colonnes d'entrée vers des noms standards."""
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
    """Ajoute les colonnes internes nécessaires à l'émargement."""
    df = df.copy()
    if "present" not in df.columns:
        df["present"] = False
    if "checkin_time" not in df.columns:
        df["checkin_time"] = ""
    if "checkin_by" not in df.columns:
        df["checkin_by"] = ""
    return df

def make_base_id(row: pd.Series) -> str:
    """Base d'identité (peut être non unique si doublons)."""
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
    """Charge l'Excel, standardise, sécurise, crée un ID unique."""
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    df = standardize_columns(df)
    df = ensure_internal_columns(df)

    # Nettoyage léger
    df = df.fillna("")

    # ID unique garanti (évite StreamlitDuplicateElementKey)
    # -> base_id + index de ligne (stable dans le fichier importé)
    df["__base_id"] = df.apply(make_base_id, axis=1)
    df["__id"] = df["__base_id"] + "|row:" + df.index.astype(str)
    df = df.drop(columns=["__base_id"], errors="ignore")

    return df

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Réglages")
    staff_name = st.text_input(
        "Nom de l'agent (optionnel)",
        placeholder="Ex: Ambroise"
    ).strip()

    st.caption("Le nom de l’agent sera enregistré dans la colonne checkin_by.")

# -------------------------
# Upload
# -------------------------
uploaded = st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("➡️ Importe un fichier Excel pour démarrer.")
    st.stop()

# Charger une seule fois par fichier (évite relecture à chaque clic)
if "df" not in st.session_state or st.session_state.get("filename") != uploaded.name:
    st.session_state.df = load_excel(uploaded)
    st.session_state.filename = uploaded.name

df = st.session_state.df

# -------------------------
# KPIs
# -------------------------
total = len(df)
present_count = int(df["present"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("Participants", total)
c2.metric("Présents", present_count)
c3.metric("Restants", total - present_count)
st.divider()

# -------------------------
# Recherche / filtres
# -------------------------
left, right = st.columns([3, 1])
with left:
    query = st.text_input("Recherche", placeholder="Nom, prénom, email, société…").strip().lower()
with right:
    only_not_present = st.checkbox("Non émargés", value=False)

# Colonnes à afficher (si elles existent)
display_cols = [c for c in STANDARD_ORDER if c in df.columns]
# fallback si Excel ne match pas nos alias
if not display_cols:
    display_cols = [c for c in df.columns if c not in ["present", "checkin_time", "checkin_by", "__id"]][:4]

search_cols = list(dict.fromkeys(display_cols + [c for c in ["email", "company", "function"] if c in df.columns]))

# Filtrage
view = df.copy()
if query:
    mask = view.apply(lambda r: query in build_search_text(r, search_cols), axis=1)
    view = view[mask].copy()

if only_not_present:
    view = view[view["present"] == False].copy()

# Tri (si possible)
if "last_name" in view.columns and "first_name" in view.columns:
    view = view.sort_values(by=["last_name", "first_name"], kind="stable")
elif "last_name" in view.columns:
    view = view.sort_values(by=["last_name"], kind="stable")

# Limite affichage (évite lag si gros fichier)
MAX_ROWS = 400
if len(view) > MAX_ROWS:
    st.warning(f"{len(view)} résultats. Affichage limité à {MAX_ROWS}. Affine la recherche.")
    view = view.head(MAX_ROWS)

# -------------------------
# Liste : 1 participant = 1 ligne
# -------------------------
st.subheader("Liste des participants")

# En-tête
h = st.columns([2, 2, 3, 3, 3, 2, 2])
h[0].markdown("**Prénom**")
h[1].markdown("**Nom**")
h[2].markdown("**Email**")
h[3].markdown("**Société**")
h[4].markdown("**Fonction**")
h[5].markdown("**Statut**")
h[6].markdown("**Action**")

# Lignes
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

export_df = df.drop(columns=["__id"], errors="ignore").copy()

st.download_button(
    "Télécharger le CSV d’émargement",
    export_df.to_csv(index=False).encode("utf-8"),
    file_name="emargement_export.csv",
    mime="text/csv",
)

with st.expander("Voir uniquement les présents"):
    st.dataframe(export_df[export_df["present"] == True], use_container_width=True)
