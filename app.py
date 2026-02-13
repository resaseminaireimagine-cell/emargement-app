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
    s = s.replace("\u00A0", " ")
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

    # base_id stable (pour log / restauration) + id unique pour UI
    df["__base_id"] = df.apply(make_base_id, axis=1)
    df["__id"] = df["__base_id"] + "|row:" + df.index.astype(str)
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

    csv_all = export_df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    present_only = export_df[export_df["present"] == True].copy()
    csv_present = present_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Emargement")
        present_only.to_excel(writer, index=False, sheet_name="Presents")
    xlsx = buffer.getvalue()

    return csv_all, csv_present, xlsx


def build_log_csv(log_rows: list[dict]) -> bytes:
    if not log_rows:
        df_log = pd.DataFrame(columns=["timestamp", "action", "base_id", "agent"])
    else:
        df_log = pd.DataFrame(log_rows)
    return df_log.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")


def apply_log_to_df(df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique un LOG à un DF importé : on garde la dernière action par base_id.
    """
    df = df.copy()
    if log_df.empty:
        return df

    # trouver colonnes log
    def find_col(target: str, variants: list[str]):
        for c in log_df.columns:
            if norm(c) == target:
                return c
        for v in variants:
            for c in log_df.columns:
                if norm(c) == v:
                    return c
        return None

    c_ts = find_col("timestamp", ["time", "date", "datetime", "heure"])
    c_action = find_col("action", ["event", "type"])
    c_base = find_col("base_id", ["id", "participant_id"])
    c_agent = find_col("agent", ["by", "user", "checkin_by"])

    if not c_action or not c_base:
        return df

    tmp = log_df.copy()
    if c_ts:
        tmp["_ts"] = pd.to_datetime(tmp[c_ts], errors="coerce")
        tmp = tmp.sort_values(by=["_ts"], kind="stable")
    tmp = tmp.dropna(subset=[c_base])
    last = tmp.groupby(tmp[c_base]).tail(1)

    by_base = df.set_index("__base_id", drop=False)

    for _, r in last.iterrows():
        base_id = str(r[c_base])
        action = str(r[c_action]).strip().lower()
        ts = str(r[c_ts]) if c_ts else ""
        agent = str(r[c_agent]) if c_agent else ""

        if base_id not in by_base.index:
            continue

        if action in ["emarger", "émarger", "checkin", "present", "présent", "in"]:
            by_base.loc[base_id, "present"] = True
            by_base.loc[base_id, "checkin_time"] = ts
            by_base.loc[base_id, "checkin_by"] = agent
        elif action in ["annuler", "cancel", "out", "remove"]:
            by_base.loc[base_id, "present"] = False
            by_base.loc[base_id, "checkin_time"] = ""
            by_base.loc[base_id, "checkin_by"] = ""

    return by_base.reset_index(drop=True)


def pager(page_count: int, page_value: int, label: str):
    c_prev, c_info, c_next = st.columns([1, 2, 1], vertical_alignment="center")
    with c_prev:
        if st.button("⬅️ Page précédente", disabled=(page_value <= 1), key=f"prev_{label}", use_container_width=True):
            st.session_state.page = max(1, page_value - 1)
            st.rerun()
    with c_info:
        st.markdown(
            f"<div style='text-align:center; font-weight:800; padding:0.35rem 0;'>Page {page_value} / {page_count}</div>",
            unsafe_allow_html=True,
        )
    with c_next:
        if st.button("Page suivante ➡️", disabled=(page_value >= page_count), key=f"next_{label}", use_container_width=True):
            st.session_state.page = min(page_count, page_value + 1)
            st.rerun()


def relevance_score(row: pd.Series, q: str) -> int:
    if not q:
        return 0
    q = q.strip().lower()

    fn = str(row.get("first_name", "")).strip().lower()
    ln = str(row.get("last_name", "")).strip().lower()
    em = str(row.get("email", "")).strip().lower()
    co = str(row.get("company", "")).strip().lower()
    fu = str(row.get("function", "")).strip().lower()

    score = 0
    if q == em:
        score += 200
    if q == ln:
        score += 140
    if q == fn:
        score += 120

    if ln.startswith(q):
        score += 110
    if fn.startswith(q):
        score += 90
    if em.startswith(q):
        score += 80
    if co.startswith(q):
        score += 50

    if q in ln:
        score += 60
    if q in fn:
        score += 45
    if q in em:
        score += 35
    if q in co:
        score += 25
    if q in fu:
        score += 10

    if not bool(row.get("present", False)):
        score += 5

    return score


# =========================
# PAGE CONFIG + CSS
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")

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

/* Boutons */
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

/* Badges */
.badge-present {{
  background:#DCFCE7; color:#166534; padding:6px 12px; border-radius:10px; font-weight:800;
  display:inline-block; white-space:nowrap;
}}
.badge-todo {{
  background:#F3F4F6; color:#374151; padding:6px 12px; border-radius:10px; font-weight:800;
  display:inline-block; white-space:nowrap;
}}

/* Noms complets (PAS de coupure / PAS d'ellipsis) */
.cell-nowrap {{
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: unset !important;
}}

@media (max-width: 980px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}

  /* Tablette : textes badges/boutons un peu plus petits pour éviter la coupure */
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
    st.caption("Fuseau horaire : **Europe/Paris** ✅")
    st.markdown("---")
    st.subheader("Sécurité")
    st.caption("Télécharge un LOG pendant l’événement (ou à la fin) pour pouvoir restaurer l’état en cas de plantage.")


# =========================
# UPLOAD + RESTORE
# =========================
uploaded = st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("➡️ Importez un fichier Excel pour commencer.")
    st.stop()

if "df" not in st.session_state or st.session_state.get("filename") != uploaded.name:
    st.session_state.df = load_excel(uploaded)
    st.session_state.filename = uploaded.name
    st.session_state.page = 1
    st.session_state["_prev_query"] = ""
    st.session_state.log = []

df = st.session_state.df
if "log" not in st.session_state:
    st.session_state.log = []

# Restore depuis log (optionnel)
with st.expander("🛟 Restaurer depuis un LOG (optionnel)"):
    st.caption("En cas de crash : ré-importe ton Excel puis uploade le LOG téléchargé pendant l’événement.")
    log_file = st.file_uploader("Importer un fichier LOG (.csv)", type=["csv"], key="log_restore")
    if log_file is not None:
        try:
            log_df = pd.read_csv(log_file, sep=";", encoding="utf-8-sig")
            restored = apply_log_to_df(df, log_df)
            # recréer __id basé sur nouvel index
            restored["__id"] = restored["__base_id"] + "|row:" + restored.index.astype(str)
            st.session_state.df = restored
            df = restored
            st.success("Restauration appliquée ✅")
            st.session_state.page = 1
        except Exception as e:
            st.error("Impossible de lire / appliquer ce LOG.")
            st.caption(repr(e))

# =========================
# SECURITY DOWNLOADS
# =========================
csv_all, csv_present, xlsx_all = build_exports(df)
log_bytes = build_log_csv(st.session_state.log)

s1, s2, s3 = st.columns([1, 1, 1], vertical_alignment="center")
with s1:
    st.download_button("⬇️ Télécharger LOG", data=log_bytes, file_name="emargement_log.csv",
                       mime="text/csv", use_container_width=True)
with s2:
    st.download_button("⬇️ Backup Excel", data=xlsx_all, file_name="emargement_backup.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
with s3:
    st.download_button("⬇️ CSV (présents)", data=csv_present, file_name="emargement_presents.csv",
                       mime="text/csv", use_container_width=True)

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
# KPI + SEARCH
# =========================
k1, k2, k3, k4 = st.columns([1, 1, 1, 2], vertical_alignment="center")
k1.metric("Participants", total)
k2.metric("Présents", present_count)
k3.metric("Restants", remaining)

with k4:
    query = st.text_input(
        "Recherche",
        placeholder="Nom, prénom, email, société…",
        key="search_query",
    ).strip().lower()

# reset page si recherche change
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
    st.caption("Astuce : la recherche remonte automatiquement les meilleurs résultats.")

st.divider()

# =========================
# VIEW FILTER + RELEVANCE SORT
# =========================
display_cols = [c for c in STANDARD_ORDER if c in df.columns]
if not display_cols:
    display_cols = [c for c in df.columns if c not in ["present", "checkin_time", "checkin_by", "__id", "__base_id"]][:4]

search_cols = list(dict.fromkeys(display_cols + [c for c in ["email", "company", "function"] if c in df.columns]))

view = df.copy()

if query:
    mask = view.apply(lambda r: query in search_text(r, search_cols), axis=1)
    view = view[mask].copy()
    # tri par pertinence
    view["_score"] = view.apply(lambda r: relevance_score(r, query), axis=1)
    view = view.sort_values(by=["_score"], ascending=False, kind="stable")
else:
    # tri alphabétique quand pas de recherche
    if "last_name" in view.columns and "first_name" in view.columns:
        view = view.sort_values(by=["last_name", "first_name"], kind="stable")

if show_present_only:
    view = view[view["present"] == True].copy()
elif only_not_present:
    view = view[view["present"] == False].copy()

# auto-cible : si 1 seul résultat non émargé
auto_target_id = None
if query:
    candidates = view[view["present"] == False]
    if len(candidates) == 1:
        auto_target_id = candidates.iloc[0]["__id"]

# =========================
# QUICK TARGET CARD
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
              <div style="font-weight:900;font-size:1.15rem; white-space:nowrap;">
                {target_row.get("first_name","")} {target_row.get("last_name","")}
              </div>
              <div style="color:{MUTED};margin-top:2px; white-space:nowrap;">
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
            idx = df.index[df["__id"] == auto_target_id]
            if len(idx):
                i = idx[0]
                base_id = df.at[i, "__base_id"]
                df.at[i, "present"] = True
                df.at[i, "checkin_time"] = now_paris_str()
                df.at[i, "checkin_by"] = staff_name
                st.session_state.df = df
                st.session_state.log.append({
                    "timestamp": now_paris_str(),
                    "action": "emarger",
                    "base_id": base_id,
                    "agent": staff_name
                })
            st.rerun()

    st.divider()

# =========================
# PAGINATION
# =========================
PAGE_SIZE = 25 if tablet_mode else 50
total_rows = len(view)
page_count = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

if "page" not in st.session_state:
    st.session_state.page = 1
st.session_state.page = min(max(1, st.session_state.page), page_count)

# pagination haut
pager(page_count, st.session_state.page, label="top")

start = (st.session_state.page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
view_page = view.iloc[start:end].copy()

# =========================
# LIST
# =========================
st.subheader("Liste des participants")

header = st.columns([2.5, 3, 3, 3, 3, 2, 2], vertical_alignment="center")
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
    base_id = row["__base_id"]
    is_present = bool(row["present"])

    fn = row.get("first_name", "")
    ln = row.get("last_name", "")
    em = row.get("email", "")
    co = row.get("company", "")
    fu = row.get("function", "")

    cols = st.columns([2.5, 3, 3, 3, 3, 2, 2], vertical_alignment="center")

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
                st.session_state.log.append({
                    "timestamp": now_paris_str(),
                    "action": "emarger",
                    "base_id": base_id,
                    "agent": staff_name
                })
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
                st.session_state.log.append({
                    "timestamp": now_paris_str(),
                    "action": "annuler",
                    "base_id": base_id,
                    "agent": staff_name
                })
            st.rerun()

# pagination bas
pager(page_count, st.session_state.page, label="bottom")

st.caption(f"Affichage : {start+1}-{min(end, total_rows)} / {total_rows}")
