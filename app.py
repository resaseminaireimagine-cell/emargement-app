import io
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import altair as alt

# Email (optionnel, activé uniquement si SMTP est configuré dans Secrets)
import smtplib
from email.message import EmailMessage


# =========================
# CONFIG
# =========================
APP_TITLE = "Outil d’émargement — Institut Imagine"
PRIMARY = "#C4007A"  # rose Imagine
BG = "#F6F7FB"
TEXT = "#111827"
MUTED = "#6B7280"

DEFAULT_TO_EMAIL = "evenements@institutimagine.org"

LOGO_CANDIDATES = [
    "logo_rose.png",
    "LOGO ROSE.png",
    "LOGO_ROSE.png",
    "logo.png",
]

# Excel column mapping
ALIASES = {
    "first_name": ["first_name", "firstname", "prenom", "prénom", "given name", "given_name"],
    "last_name":  ["last_name", "lastname", "nom", "surname", "family name", "family_name"],
    "email":      ["email", "e-mail", "mail", "courriel"],
    "company":    ["company", "societe", "société", "organisation", "organization", "structure"],
    "function":   ["fonction", "function", "job", "poste", "title"],
}
STANDARD_ORDER = ["first_name", "last_name", "email", "company", "function"]


# =========================
# HELPERS
# =========================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


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
    co = str(row.get("company", "")).strip().lower()
    return f"name:{ln}|{fn}|{co}"


def load_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    df = standardize_columns(df)
    df = ensure_internal_columns(df)
    df = df.fillna("")

    # ID unique par ligne => évite StreamlitDuplicateElementKey
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

    # CSV Excel FR = séparateur ;
    csv_all = export_df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    present_only = export_df[export_df["present"] == True].copy()
    csv_present = present_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Emargement")
        present_only.to_excel(writer, index=False, sheet_name="Presents")
    xlsx = buffer.getvalue()

    return csv_all, csv_present, xlsx


def smtp_config_available() -> bool:
    required = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "SMTP_FROM"]
    return all(k in st.secrets for k in required)


def send_email_with_attachment(to_addr: str, subject: str, body: str, attachment_bytes: bytes, filename: str) -> None:
    msg = EmailMessage()
    msg["From"] = st.secrets["SMTP_FROM"]
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    msg.add_attachment(
        attachment_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
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
# PAGE CONFIG
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")

# CSS (tablette + contrastes)
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
}}

button[kind="secondary"], .stButton > button[kind="secondary"] {{
  background: #ffffff !important;
  color: {PRIMARY} !important;
  border: 2px solid {PRIMARY} !important;
}}

@media (max-width: 980px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  .stButton > button {{ min-height: 52px !important; font-size: 1.05rem !important; }}
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
        st.image(logo_path, width=110)
with c2:
    st.markdown(f"## {APP_TITLE}")
    st.caption("Importez votre liste, recherchez un participant, émargez, puis exportez / envoyez la feuille d’émargement.")

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
    to_email = st.text_input("Email de destination", value=DEFAULT_TO_EMAIL).strip()

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

df = st.session_state.df

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

f1, f2, f3 = st.columns([1, 1, 2], vertical_alignment="center")
with f1:
    only_not_present = st.checkbox("Non émargés", value=True)
with f2:
    show_present_only = st.checkbox("Présents uniquement", value=False)
with f3:
    st.caption("Affichage optimisé : 1 ligne = 1 participant")

# =========================
# DASHBOARD / CHARTS
# =========================
st.subheader("Tableau de bord")

d1, d2 = st.columns([1, 1], vertical_alignment="top")

with d1:
    progress_df = pd.DataFrame({"Statut": ["Présents", "Restants"], "Nombre": [present_count, remaining]})
    donut = (
        alt.Chart(progress_df)
        .mark_arc(innerRadius=70)
        .encode(
            theta=alt.Theta("Nombre:Q"),
            color=alt.Color("Statut:N", legend=alt.Legend(title=None)),
            tooltip=["Statut:N", "Nombre:Q"],
        )
        .properties(height=220)
    )
    st.altair_chart(donut, use_container_width=True)

with d2:
    tmp = df[df["present"] == True].copy()
    tmp["checkin_time_dt"] = pd.to_datetime(tmp["checkin_time"], errors="coerce")
    tmp = tmp.dropna(subset=["checkin_time_dt"])
    if len(tmp):
        tmp["minute"] = tmp["checkin_time_dt"].dt.floor("min")
        t = tmp.groupby("minute").size().reset_index(name="Arrivées").sort_values("minute")
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
        st.caption("Aucun émargement pour l’instant (le graphique apparaîtra dès les premières arrivées).")

if "company" in df.columns:
    st.caption("Top sociétés (présents)")
    tmp = df[df["present"] == True].copy()
    if len(tmp):
        top = tmp.groupby("company").size().reset_index(name="Présents").sort_values("Présents", ascending=False).head(10)
        bar = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                y=alt.Y("company:N", sort="-x", title=None),
                x=alt.X("Présents:Q", title="Présents"),
                tooltip=["company:N", "Présents:Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.caption("Pas encore de présents → le top sociétés s’affichera ensuite.")

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
page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
start = (page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
view_page = view.iloc[start:end].copy()

st.subheader("Liste des participants")

# Header row
h = st.columns([2, 2, 3, 3, 3, 2, 2])
h[0].markdown("**Prénom**")
h[1].markdown("**Nom**")
h[2].markdown("**Email**")
h[3].markdown("**Société**")
h[4].markdown("**Fonction**")
h[5].markdown("**Statut**")
h[6].markdown("**Action**")

def badge(is_present: bool) -> str:
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
    cols[5].markdown(badge(is_present), unsafe_allow_html=True)

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
# EXPORTS + EMAIL
# =========================
st.subheader("Exports")

csv_all, csv_present, xlsx_all = build_exports(df)

e1, e2, e3 = st.columns([1, 1, 1], vertical_alignment="center")
with e1:
    st.download_button("⬇️ CSV (Excel FR)", data=csv_all, file_name="emargement_export.csv",
                       mime="text/csv", use_container_width=True)
with e2:
    st.download_button("⬇️ CSV (présents)", data=csv_present, file_name="emargement_presents.csv",
                       mime="text/csv", use_container_width=True)
with e3:
    st.download_button("⬇️ Excel (.xlsx)", data=xlsx_all, file_name="emargement_export.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

st.markdown("### Envoi par email")
st.caption("Le bouton s’active uniquement si la configuration SMTP est renseignée dans Streamlit Secrets.")

subject_default = f"[Institut Imagine] Export émargement — {datetime.now().strftime('%d/%m/%Y %H:%M')}"
email_subject = st.text_input("Objet", value=subject_default)
email_body = st.text_area(
    "Message",
    value="Bonjour,\n\nVeuillez trouver en pièce jointe l’export d’émargement (Excel).\n\nCordialement,\nInstitut Imagine",
    height=120,
)

can_send = smtp_config_available()
if st.button("📧 Envoyer l’export à evenements@institutimagine.org", use_container_width=True, disabled=not can_send):
    try:
        send_email_with_attachment(
            to_addr=to_email or DEFAULT_TO_EMAIL,
            subject=email_subject,
            body=email_body,
            attachment_bytes=xlsx_all,
            filename="emargement_export.xlsx",
        )
        st.success(f"Email envoyé à {to_email or DEFAULT_TO_EMAIL}")
    except Exception as e:
        st.error("Échec de l’envoi email. Vérifiez la config SMTP dans Secrets.")
        st.caption(str(e))

if not can_send:
    with st.expander("Configurer l’envoi email (SMTP)"):
        st.code(
            'SMTP_HOST="smtp.votre-domaine.org"\n'
            'SMTP_PORT="587"\n'
            'SMTP_USER="compte_smtp@institutimagine.org"\n'
            'SMTP_PASS="mot_de_passe_ou_app_password"\n'
            'SMTP_FROM="evenements@institutimagine.org"\n',
            language="toml",
        )
