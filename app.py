# app.py — Outil d’émargement — Institut Imagine
# Version finale (sans dépendance streamlit-keyup) — recherche instantanée, export CSV/Excel, pagination bas, heure FR.

from __future__ import annotations

import io
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Outil d’émargement — Institut Imagine"
TZ = ZoneInfo("Europe/Paris")
ACCENT = "#C4007A"  # rose Institut Imagine (approx.)
LOGO_PATH = "LOGO ROSE.png"  # fichier présent dans ton repo
PAGE_SIZE_DEFAULT = 25
PAGE_SIZE_OPTIONS = [10, 25, 50, 100]

# -----------------------------
# Helpers
# -----------------------------
def now_paris_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def normalize_colname(s: str) -> str:
    s = str(s).strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns + map common variants to standard schema."""
    df = df.copy()

    # Normalize headers
    df.columns = [normalize_colname(c) for c in df.columns]

    # Common mappings -> standard names
    mappings = {
        # first name
        "firstname": "first_name",
        "first": "first_name",
        "prenom": "first_name",
        "prénom": "first_name",
        "given_name": "first_name",
        "first_name": "first_name",
        # last name
        "lastname": "last_name",
        "last": "last_name",
        "nom": "last_name",
        "surname": "last_name",
        "family_name": "last_name",
        "last_name": "last_name",
        # email
        "mail": "email",
        "e_mail": "email",
        "email": "email",
        # company
        "societe": "company",
        "société": "company",
        "organisation": "company",
        "organization": "company",
        "company": "company",
        "employer": "company",
        # role
        "fonction": "role",
        "poste": "role",
        "job_title": "role",
        "title": "role",
        "role": "role",
        # status
        "present": "present",
        "présent": "present",
        "presence": "present",
        "présence": "present",
        # timestamps
        "heure": "checkin_time",
        "checkin_time": "checkin_time",
        "check_in_time": "checkin_time",
        "checkin": "checkin_time",
        "check_in": "checkin_time",
        # who
        "checkin_by": "checkin_by",
        "checked_in_by": "checkin_by",
        "agent": "checkin_by",
        "operateur": "checkin_by",
        "operateur_checkin": "checkin_by",
    }

    # Apply mappings
    new_cols = []
    for c in df.columns:
        new_cols.append(mappings.get(c, c))
    df.columns = new_cols

    # Keep only needed columns (but preserve extra cols if you want)
    for required in ["first_name", "last_name"]:
        if required not in df.columns:
            df[required] = ""

    if "email" not in df.columns:
        df["email"] = ""
    if "company" not in df.columns:
        df["company"] = ""
    if "role" not in df.columns:
        df["role"] = ""

    # Status columns
    if "present" not in df.columns:
        df["present"] = False
    else:
        # Normalize present values
        df["present"] = df["present"].apply(
            lambda x: True
            if str(x).strip().lower() in ["true", "1", "yes", "y", "oui", "present", "présent"]
            else False
        )

    if "checkin_time" not in df.columns:
        df["checkin_time"] = ""
    else:
        df["checkin_time"] = df["checkin_time"].fillna("").astype(str)

    if "checkin_by" not in df.columns:
        df["checkin_by"] = ""
    else:
        df["checkin_by"] = df["checkin_by"].fillna("").astype(str)

    # Clean NaN in key fields
    for c in ["first_name", "last_name", "email", "company", "role"]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    # Add internal unique id (stable-ish)
    # Use existing index + normalized name/email to reduce duplicate key collisions
    df["_rid"] = (
        df.index.astype(str)
        + "|"
        + df["first_name"].apply(lambda s: _strip_accents(s).lower())
        + "|"
        + df["last_name"].apply(lambda s: _strip_accents(s).lower())
        + "|"
        + df["email"].apply(lambda s: s.lower())
    )

    # Search field (accent-insensitive)
    def build_search(row) -> str:
        parts = [
            row.get("first_name", ""),
            row.get("last_name", ""),
            row.get("email", ""),
            row.get("company", ""),
            row.get("role", ""),
        ]
        joined = " ".join([p for p in parts if p])
        joined = _strip_accents(joined).lower()
        return joined

    df["_search"] = df.apply(build_search, axis=1)

    return df


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    export = df.copy()

    # Clean internal columns
    export = export.drop(columns=[c for c in export.columns if c.startswith("_")], errors="ignore")

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export.to_excel(writer, index=False, sheet_name="Emargement")
    buf.seek(0)
    return buf.read()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    export = df.copy()
    export = export.drop(columns=[c for c in export.columns if c.startswith("_")], errors="ignore")
    # Excel FR friendly: separator ;
    return export.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")


def inject_css():
    st.markdown(
        f"""
<style>
/* Global */
html, body, [class*="css"] {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}}
/* Page background */
.main {{
  background: #f6f7fb;
}}
/* Header card */
.im-header {{
  background: white;
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
  display:flex;
  gap:18px;
  align-items:center;
}}
.im-title {{
  font-size: 34px;
  font-weight: 800;
  margin: 0;
  line-height: 1.1;
}}
.im-sub {{
  margin: 6px 0 0 0;
  color: #6b7280;
  font-size: 14px;
}}
/* KPI cards row */
.kpi {{
  background: white;
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}}
.kpi-label {{
  color: #6b7280;
  font-size: 14px;
  margin-bottom: 6px;
}}
.kpi-value {{
  font-size: 44px;
  font-weight: 800;
  line-height: 1;
}}
/* Filter bar */
.filters {{
  background: white;
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}}
/* Table card row */
.rowcard {{
  background: white;
  border-radius: 18px;
  padding: 10px 14px;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
  margin-bottom: 12px;
}}
/* Status pill */
.pill {{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 6px 10px;
  border-radius: 999px;
  background: #f3f4f6;
  color: #111827;
  font-weight: 600;
  font-size: 14px;
  white-space: nowrap;
}}
.pill.ok {{
  background: #ecfdf5;
  color: #065f46;
}}
/* Buttons */
div.stButton > button {{
  border-radius: 14px !important;
  border: 2px solid {ACCENT} !important;
  padding: 10px 14px !important;
  font-weight: 800 !important;
  white-space: nowrap !important;
}}
/* Primary action in row: force white text */
button[kind="secondary"] {{
  /* leave default */
}}
/* Streamlit sometimes sets inline colors; we ensure contrast via filter for our accent buttons */
.im-accent button {{
  background: {ACCENT} !important;
  color: white !important;
}}
.im-accent button * {{
  color: white !important;
}}
/* Search input look */
div[data-baseweb="input"] > div {{
  border-radius: 14px !important;
}}
/* Tablet tweaks */
@media (max-width: 900px) {{
  .im-title {{ font-size: 26px; }}
  .kpi-value {{ font-size: 34px; }}
  .pill {{ font-size: 12px; padding: 5px 8px; }}
  div.stButton > button {{ padding: 9px 10px !important; font-size: 13px !important; }}
}}
/* Prevent last name wrapping: we keep it on one line and allow horizontal scroll inside cell if needed */
.nowrap {{
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
</style>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

inject_css()

# Sidebar settings
with st.sidebar:
    st.markdown("## Réglages")
    agent_name = st.text_input("Nom de l’agent (optionnel)", placeholder="Ex: Ambroise", key="agent_name")
    st.caption("Ce nom est enregistré dans la colonne **checkin_by**.")
    st.markdown("---")
    st.caption("Astuce Excel : l’export CSV utilise **;** (compatibilité Excel FR).")

# Header
c1, c2 = st.columns([1, 5])
with c1:
    try:
        st.image(LOGO_PATH, use_container_width=True)
    except Exception:
        st.write("")

with c2:
    st.markdown(
        f"""
<div class="im-header">
  <div style="flex:1;">
    <div class="im-title">{APP_TITLE}</div>
    <div class="im-sub">Importez votre liste, recherchez un participant, émargez, puis exportez la feuille d’émargement.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")

# Upload
uploaded = st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False)

if "df" not in st.session_state:
    st.session_state.df = None

if uploaded is not None:
    try:
        raw = pd.read_excel(uploaded)
        df = ensure_columns(raw)

        # Store once per upload (reset pagination & filters)
        st.session_state.df = df
        st.session_state.page = 1
    except Exception as e:
        st.error("Impossible de lire ce fichier Excel. Vérifiez qu’il s’agit bien d’un .xlsx valide.")
        st.stop()

df = st.session_state.df

if df is None:
    st.info("➡️ Importez un fichier Excel pour commencer.")
    st.stop()

# Pie chart ABOVE KPI band (camembert)
present_count = int(df["present"].sum())
total_count = int(len(df))
remaining_count = total_count - present_count

pie_df = pd.DataFrame(
    {"Statut": ["Présents", "Restants"], "Nombre": [present_count, remaining_count]}
)

# Use Altair via Streamlit's native chart (no seaborn); keep default colors (Streamlit chooses).
st.markdown("### Répartition")
st.pyplot(None)  # no-op placeholder avoidance for some environments

try:
    import altair as alt

    pie = (
        alt.Chart(pie_df)
        .mark_arc(innerRadius=55)
        .encode(theta="Nombre:Q", color="Statut:N", tooltip=["Statut:N", "Nombre:Q"])
        .properties(height=240)
    )
    st.altair_chart(pie, use_container_width=True)
except Exception:
    # If altair unavailable for any reason, fallback to text
    st.caption("Graphique indisponible (altair).")

# KPI + search
k1, k2, k3, k4 = st.columns([1, 1, 1, 2])
with k1:
    st.markdown(
        f"""
<div class="kpi">
  <div class="kpi-label">Participants</div>
  <div class="kpi-value">{total_count}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
<div class="kpi">
  <div class="kpi-label">Présents</div>
  <div class="kpi-value">{present_count}</div>
</div>
""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
<div class="kpi">
  <div class="kpi-label">Restants</div>
  <div class="kpi-value">{remaining_count}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with k4:
    # IMPORTANT: no st.form() here -> filtering is instant; no Enter required
    search = st.text_input(
        "Recherche",
        placeholder="Nom, prénom, email, société…",
        key="search_input",
    )
    search = (search or "").strip().lower()
    search = _strip_accents(search)

# Filters
st.write("")
fcol = st.columns([1, 1, 3])
with fcol[0]:
    only_not_checked = st.checkbox("Non émargés", value=False)
with fcol[1]:
    only_present = st.checkbox("Présents uniquement", value=False)
with fcol[2]:
    st.caption("La recherche remonte automatiquement les meilleurs résultats.")

# Export buttons
st.write("")
ex1, ex2, ex3 = st.columns([1, 1, 2])
export_df = df.copy().drop(columns=["_search"], errors="ignore")

with ex1:
    st.download_button(
        "⬇️ Export Excel",
        data=to_excel_bytes(export_df),
        file_name="emargement_institut_imagine.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with ex2:
    st.download_button(
        "⬇️ Export CSV (Excel FR)",
        data=to_csv_bytes(export_df),
        file_name="emargement_institut_imagine.csv",
        mime="text/csv",
        use_container_width=True,
    )
with ex3:
    page_size = st.selectbox("Participants par page", PAGE_SIZE_OPTIONS, index=PAGE_SIZE_OPTIONS.index(PAGE_SIZE_DEFAULT))

# Apply filters + search
view = df.copy()

if only_not_checked:
    view = view[view["present"] == False]  # noqa: E712
if only_present:
    view = view[view["present"] == True]  # noqa: E712

if search:
    view = view[view["_search"].str.contains(search, na=False)]

# Pagination
if "page" not in st.session_state:
    st.session_state.page = 1

total_rows = len(view)
total_pages = max(1, (total_rows + page_size - 1) // page_size)

# Clamp page if needed
if st.session_state.page > total_pages:
    st.session_state.page = total_pages
if st.session_state.page < 1:
    st.session_state.page = 1

start = (st.session_state.page - 1) * page_size
end = start + page_size
page_view = view.iloc[start:end].copy()

# Pagination controls (TOP)
st.write("")
p1, p2, p3 = st.columns([2, 1, 2])
with p1:
    if st.button("⬅️ Page précédente", use_container_width=True, disabled=(st.session_state.page <= 1)):
        st.session_state.page -= 1
        st.rerun()
with p2:
    st.markdown(f"<div style='text-align:center; font-weight:800; padding-top:10px;'>Page {st.session_state.page} / {total_pages}</div>", unsafe_allow_html=True)
with p3:
    if st.button("Page suivante ➡️", use_container_width=True, disabled=(st.session_state.page >= total_pages)):
        st.session_state.page += 1
        st.rerun()

st.write("")
st.markdown("## Liste des participants")

# Table header
h = st.columns([1.1, 1.2, 1.6, 1.3, 1.2, 1.0, 1.0])
headers = ["Prénom", "Nom", "Email", "Société", "Fonction", "Statut", "Action"]
for col, txt in zip(h, headers):
    with col:
        st.markdown(f"**{txt}**")

st.write("")

# Rows
# Note: we modify st.session_state.df via _rid lookup, not the filtered view.
df_master = st.session_state.df

def set_presence(rid: str, present: bool):
    idx = df_master.index[df_master["_rid"] == rid]
    if len(idx) == 0:
        return
    i = idx[0]
    df_master.at[i, "present"] = present
    if present:
        df_master.at[i, "checkin_time"] = now_paris_str()
        df_master.at[i, "checkin_by"] = (agent_name or "").strip()
    else:
        df_master.at[i, "checkin_time"] = ""
        df_master.at[i, "checkin_by"] = ""
    st.session_state.df = df_master


for _, row in page_view.iterrows():
    rid = row["_rid"]
    cols = st.columns([1.1, 1.2, 1.6, 1.3, 1.2, 1.0, 1.0])

    # Values
    first = row.get("first_name", "")
    last = row.get("last_name", "")
    email = row.get("email", "")
    company = row.get("company", "")
    role = row.get("role", "")
    is_present = bool(row.get("present", False))

    with cols[0]:
        st.markdown(f"<div class='nowrap'>{first}</div>", unsafe_allow_html=True)
    with cols[1]:
        # Ensure full name visible as much as possible, but no wrap.
        # On small screens it may ellipsize; user can still export full value.
        st.markdown(f"<div class='nowrap' title='{last}'>{last}</div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"<div class='nowrap' title='{email}'>{email}</div>", unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"<div class='nowrap' title='{company}'>{company}</div>", unsafe_allow_html=True)
    with cols[4]:
        st.markdown(f"<div class='nowrap' title='{role}'>{role}</div>", unsafe_allow_html=True)

    with cols[5]:
        if is_present:
            st.markdown("<span class='pill ok'>✅ Présent</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='pill'>⬜ À émarger</span>", unsafe_allow_html=True)

    with cols[6]:
        if is_present:
            st.markdown("<div class='im-accent'>", unsafe_allow_html=True)
            if st.button("Annuler", key=f"undo_{rid}", use_container_width=True):
                set_presence(rid, False)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='im-accent'>", unsafe_allow_html=True)
            if st.button("Émarger", key=f"do_{rid}", use_container_width=True):
                set_presence(rid, True)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# Pagination controls (BOTTOM) — requested
st.write("")
b1, b2, b3 = st.columns([2, 1, 2])
with b1:
    if st.button("⬅️ Page précédente ", use_container_width=True, disabled=(st.session_state.page <= 1), key="prev_bottom"):
        st.session_state.page -= 1
        st.rerun()
with b2:
    st.markdown(f"<div style='text-align:center; font-weight:800; padding-top:10px;'>Page {st.session_state.page} / {total_pages}</div>", unsafe_allow_html=True)
with b3:
    if st.button("Page suivante ➡️ ", use_container_width=True, disabled=(st.session_state.page >= total_pages), key="next_bottom"):
        st.session_state.page += 1
        st.rerun()

# Small footer info (optional)
st.write("")
st.caption("Heure d’émargement : fuseau Europe/Paris. Export conseillé en fin de session.")
