import streamlit as st
import pandas as pd
from datetime import datetime
import re
import io
from pathlib import Path

# Charts
import altair as alt

# Email (optional – needs secrets)
import smtplib
from email.message import EmailMessage

# =========================
# CONFIG / BRANDING
# =========================
APP_TITLE = "Outil d’émargement — Institut Imagine"
PRIMARY = "#C4007A"   # rose Imagine
BG = "#F6F7FB"
TEXT = "#111827"
MUTED = "#6B7280"

TO_EMAIL_DEFAULT = "evenements@institutimagine.org"

st.set_page_config(page_title=APP_TITLE, layout="wide")

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

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# =========================
# CSS – pro + tablette + boutons lisibles
# =========================
st.markdown(
    f"""
<style>
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.1rem; max-width: 1280px; }}

h1, h2, h3, h4 {{ color: {TEXT}; }}
small, .stCaption, p {{ color: {MUTED}; }}

/* Cards / rows */
[data-testid="stHorizontalBlock"] {{
  background: white;
  border-radius: 16px;
  padding: 0.45rem 0.75rem;
  margin-bottom: 0.45rem;
  box-shadow: 0 1px 12px rgba(0,0,0,0.06);
}}

/* Inputs */
.stTextInput input {{
  border-radius: 14px;
  padding: 0.7rem 0.9rem;
  font-size: 1.0rem;
}}

/* Buttons – force contrast */
.stButton > button {{
  background-color: {PRIMARY} !important;
  color: white !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 0.75rem 1.05rem !important;
  font-weight: 800 !important;
  opacity: 1 !important;
  min-height: 46px !important; /* finger-friendly */
}}

.stButton > button:hover {{
  background-color: {PRIMARY} !important;
  filter: brightness(0.92);
}}

/* Secondary button style (Streamlit sets type=secondary, but we also ensure readability) */
button[kind="secondary"], .stButton > button[kind="secondary"] {{
  background: white !important;
  color: {PRIMARY} !important;
  border: 2px solid {PRIMARY} !important;
}}

@media (max-width: 980px) {{
  /* Tablet: bigger headings and tighter layout */
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  h2 {{ font-size: 1.35rem; }}
  h3 {{ font-size: 1.15rem; }}
  [data-testid="stHorizontalBlock"] {{
    padding: 0.55rem 0.65rem;
  }}
}}
</style>
""",
    unsafe_allow_html=True
)

# =========================
# Excel column mapping
# =========================
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

    # Unique ID guaranteed (avoid StreamlitDuplicateElementKey)
    df["__base_id"] = df.apply(make_base_id, axis=1)
    df["__id"] = df["__base_id"] + "|row:" + df.index.astype(str)
    df = df.drop(columns=["__base_id"], errors="ignore")
    return df

# =========================
# Exports helpers
# =========================
def build_exports(df: pd.DataFrame) -> tuple[bytes, bytes, bytes]:
    """Return (csv_all, csv_present, xlsx_all)."""
    export_df = df.drop(columns=["__id"], errors="ignore").copy()

    csv_all = export_df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    present_only = export_df[export_df["present"] == True].copy()
    csv_present = present_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Emargement")
        present_only.to_excel(writer, index=False, sheet_name="Presents")
    xlsx_all = buffer.getvalue()

    return csv_all, csv_present, xlsx_all

# =========================
# Email sending (needs Streamlit secrets)
# =========================
def smtp_config_available() -> bool:
    required = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "SMTP_FROM"]
    return all(k in st.secrets for k in required)

def send_email_with_attachment(
    to_addr: str,
    subject: str,
    body: str,
    attachment_bytes: bytes,
    attachment_filename: str,
) -> None:
    """
    Sends email via SMTP using st.secrets:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM
    """
    msg = EmailMessage()
    msg["From"] = st.secrets["SMTP_FROM"]
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    msg.add_attachment(
        attachment_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=attachment_filename,
    )

    host = st.secrets["SMTP_HOST"]
    port = int(st.secrets["SMTP_PORT"])
    user = st.secrets["SMTP_USER"]
    pwd = st.secrets["SMTP_PASS"]

    with smtplib.SMTP(host, port, timeout=20) as server:
        server.starttls()
        server.login(user, pwd)
        server.send_message(msg)

# =========================
# HEADER
# =========================
logo_path = find_logo_path()
h1, h2 = st.columns([1, 6], vertical_alignment="center")
with h1:
    if logo_path:
        st.image(logo_path, width=120)  # smaller, tablet-friendly
with h2:
    st.markdown(f"## {APP_TITLE}")
    st.caption("Importez votre liste, recherchez un participant, émargez, puis exportez / envoyez la feuille d’émargement.")

st.divider()

# =========================
# SIDEBAR (tablet: still useful, collapsible)
# =========================
with st.sidebar:
    st.header("Réglages")
    staff_name = st.text_input("Nom de l'agent (optionnel)", placeholder="Ex: Ambroise").strip()
    st.caption("Sera enregistré dans la colonne checkin_by.")
    st.markdown("---")
    tablet_mode = st.toggle("Mode tablette (touch)", value=True)
    st.caption("Grossit les boutons / champs pour usage accueil.")
    st.markdown("---")
    to_email = st.text_input("Email de destination", value=TO_EMAIL_DEFAULT).strip()
    st.caption("Bouton d’envoi en bas (nécessite config SMTP).")

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("➡️ Importez un fichier Excel pour commencer.")
    st.stop()

# Load once per filename
if "df" not in st.session_state or st.session_state.get("filename") != uploaded.name:
    st.session_state.df = load_excel(uploaded)
    st.session_state.filename = uploaded.name

df = st.session_state.df

# Tablet UX tweak (optional extra CSS)
if tablet_mode:
    st.markdown(
        """
        <style>
        .stTextInput input { font-size: 1.05rem !important; }
        .stButton > button { font-size: 1.05rem !important; min-height: 52px !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

# =========================
# KPIs + SEARCH + FILTERS
# =========================
total = len(df)
present_count = int(df["present"].sum())
remaining = total - present_count

k1, k2, k3, k4 = st.columns([1, 1, 1, 2], vertical_alignment="center")
k1.metric("Participants", total)
k2.metric("Présents", present_count)
k3.metric("Restants", remaining)
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
# CHARTS
# =========================
st.subheader("Tableau de bord")

c1, c2 = st.columns([1, 1], vertical_alignment="top")

# 1) Progress donut
with c1:
    progress_df = pd.DataFrame({
        "Statut": ["Présents", "Restants"],
        "Nombre": [present_count, remaining],
    })
    donut = (
        alt.Chart(progress_df)
        .mark_arc(innerRadius=70)
        .encode(
            theta=alt.Theta(field="Nombre", type="quantitative"),
            color=alt.Color(field="Statut", type="nominal", legend=alt.Legend(title=None)),
            tooltip=["Statut", "Nombre"],
        )
        .properties(height=220)
    )
    st.altair_chart(donut, use_container_width=True)

# 2) Check-ins over time (if any)
with c2:
    tmp = df.copy()
    tmp = tmp[tmp["present"] == True].copy()
    if len(tmp) and "checkin_time" in tmp.columns:
        tmp["checkin_time"] = pd.to_datetime(tmp["checkin_time"], errors="coerce")
        tmp = tmp.dropna(subset=["checkin_time"])
        if len(tmp):
            tmp["minute"] = tmp["checkin_time"].dt.floor("min")
            t = tmp.groupby("minute").size().reset_index(name="Arrivées")
            t = t.sort_values("minute")
            line = (
                alt.Chart(t)
                .mark_line(point=True)
                .encode(
                    x=alt.X("minute:T", title="Heure"),
                    y=alt.Y("Arrivées:Q", title="Arrivées / minute"),
                    tooltip=[alt.Tooltip("minute:T", title="Heure"), "Arrivées:Q"],
                )
                .properties(height=220)
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.caption("Aucune donnée d’heure exploitable pour le moment.")
    else:
        st.caption("Aucun émargement pour l’instant (le graphique s’affichera dès les premières arrivées).")

# Optional: Top companies (only if column exists)
if "company" in df.columns:
    st.caption("Top sociétés (présents)")
    tmp = df[df["present"] == True].copy()
    if len(tmp):
        top = (
            tmp.groupby("company")
            .size()
            .reset_index(name="Présents")
            .sort_values("Présents", ascending=False)
            .head(10)
        )
        bar = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                y=alt.Y("company:N", sort="-x", title=None),
                x=alt.X("Présents:Q", title="Présents"),
                tooltip=["company", "Présents"],
            )
            .properties(height=260)
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.caption("Pas encore de présents → le top sociétés s’affichera ensuite.")

st.divider()

# =========================
# FILTER / SORT / VIEW
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

if "last_name" in view.columns and "first_name" in view.columns:
    view = view.sort_values(by=["last_name", "first_name"], kind="stable")
elif "last_name" in view.columns:
    view = view.sort_values(by=["last_name"], kind="stable")

# Tablet-friendly pagination
PAGE_SIZE = 30 if tablet_mode else 50
total_rows = len(view)
page_count = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
start = (page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
view_page = view.iloc[start:end].copy()

st.subheader("Liste des participants")

# Header row
header = st.columns([2, 2, 3, 3, 3, 2, 2])
header[0].markdown("**Prénom**")
header[1].markdown("**Nom**")
header[2].markdown("**Email**")
header[3].markdown("**Société**")
header[4].markdown("**Fonction**")
header[5].markdown("**Statut**")
header[6].markdown("**Action**")

def status_badge(is_present: bool) -> str:
    if is_present:
        return "<span style='background:#DCFCE7;color:#166534;padding:6px 12px;border-radius:10px;font-weight:800;'>✔ Présent</span>"
    return "<span style='background:#F3F4F6;color:#374151;padding:6px 12px;border-radius:10px;font-weight:800;'>À émarger</span>"

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
    cols[5].markdown(status_badge(is_present), unsafe_allow_html=True)

    if not is_present:
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
        # Secondary (outlined) action
        if cols[6].button("Annuler", key=f"an_{rid}", use_container_width=True, type="secondary"):
            idx = df.index[df["__id"] == rid]
            if len(idx):
                i = idx[0]
                df.at[i, "present"] = False
                df.at[i, "checkin_time"] = ""
                df.at[i, "checkin_by"] = ""
                st.session_state.df = df
            st.rerun()

st.caption(f"Affichage : {start+1}-{min(end, total_rows)} / {total_rows}")

st.divider()

# =========================
# LAST CHECK-INS
# =========================
st.subheader("Derniers émargés")
last_df = df[df["present"] == True].copy()
if len(last_df):
    last_df["checkin_time_dt"] = pd.to_datetime(last_df["checkin_time"], errors="coerce")
    last_df = last_df.sort_values(by=["checkin_time_dt"], ascending=False, kind="stable").head(10)
    show_cols = [c for c in ["first_name", "last_name", "company", "checkin_time", "checkin_by"] if c in last_df.columns]
    st.dataframe(last_df.drop(columns=["__id", "checkin_time_dt"], errors="ignore")[show_cols],
                 use_container_width=True, hide_index=True)
else:
    st.caption("Aucun émargement pour l’instant.")

st.divider()

# =========================
# EXPORTS + EMAIL
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
        use_container_width=True
    )
with e2:
    st.download_button(
        "⬇️ CSV (présents)",
        data=csv_present,
        file_name="emargement_presents.csv",
        mime="text/csv",
        use_container_width=True
    )
with e3:
    st.download_button(
        "⬇️ Excel (.xlsx)",
        data=xlsx_all,
        file_name="emargement_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.markdown("### Envoi par email")
st.caption("⚠️ L’envoi automatique nécessite une configuration SMTP dans Streamlit Secrets (sinon le bouton est désactivé).")

send_col1, send_col2 = st.columns([2, 1], vertical_alignment="center")
with send_col1:
    email_subject = st.text_input(
        "Objet",
        value=f"[Institut Imagine] Export émargement — {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    )
    email_body = st.text_area(
        "Message",
        value=(
            "Bonjour,\n\n"
            "Veuillez trouver en pièce jointe l’export d’émargement (Excel).\n\n"
            "Cordialement,\n"
            "Outil d’émargement — Institut Imagine"
        ),
        height=140,
    )

with send_col2:
    can_send = smtp_config_available()
    if st.button("📧 Envoyer l’export", use_container_width=True, disabled=not can_send):
        try:
            send_email_with_attachment(
                to_addr=to_email,
                subject=email_subject,
                body=email_body,
                attachment_bytes=xlsx_all,
                attachment_filename="emargement_export.xlsx",
            )
            st.success(f"Email envoyé à {to_email}")
        except Exception as e:
            st.error("Échec de l’envoi email. Vérifiez la config SMTP dans Secrets.")
            st.caption(str(e))

if not smtp_config_available():
    with st.expander("Configurer l’envoi email (SMTP)"):
        st.markdown(
            """
Pour activer le bouton **Envoyer l’export**, ajoute ces variables dans Streamlit Cloud :

**App → Settings → Secrets** (ou `secrets.toml`) :

```toml
SMTP_HOST="smtp.votre-domaine.org"
SMTP_PORT="587"
SMTP_USER="compte_smtp@institutimagine.org"
SMTP_PASS="mot_de_passe_ou_app_password"
SMTP_FROM="evenements@institutimagine.org"
