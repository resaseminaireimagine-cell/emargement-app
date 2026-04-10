import base64
import hashlib
import io
import json
import re
import unicodedata
import urllib.parse
import zlib
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st


APP_TITLE = "EMARGEMENT"
APP_SUBTITLE = "Application de suivi des presences pour evenements et seminaires."
APP_TAGLINE = "Guerir les maladies genetiques"
APP_BUILD = "2026-04-10-02"
PRIMARY = "#D0007F"
PRIMARY_DARK = "#8B0D59"
PRIMARY_DEEP = "#650B41"
PRIMARY_SOFT = "#F7D7E8"
PRIMARY_SOFTEST = "#FDF4F9"
BG = "#FBF7FA"
SURFACE = "#FFFFFF"
BORDER = "#E8D7E2"
TEXT = "#2B1630"
MUTED = "#71566B"
PARIS_TZ = ZoneInfo("Europe/Paris")
MAIL_TO = "evenements@institutimagine.org"
LOGO_CANDIDATES = ["logo_rose.png", "LOGO ROSE.png", "LOGO_ROSE.png", "logo.png"]

ALIASES = {
    "first_name": ["first_name", "firstname", "first name", "given name", "given_name", "prenom"],
    "last_name": ["last_name", "lastname", "last name", "surname", "family name", "family_name", "nom"],
    "email": ["email", "e-mail", "mail", "courriel"],
    "company": ["company", "societe", "organisation", "organization", "structure"],
    "function": ["fonction", "function", "job", "poste", "title"],
    "present": ["present", "presence", "statut"],
    "checkin_time": ["checkin_time", "checkin time", "heure", "date", "datetime", "check-in time"],
    "checkin_by": ["checkin_by", "checkin by", "agent", "emarge par", "checked in by"],
}

DISPLAY_ORDER = ["first_name", "last_name", "email", "company"]
TEXT_COLUMNS = [
    "first_name",
    "last_name",
    "email",
    "company",
    "function",
    "checkin_time",
    "checkin_by",
]
PRESENT_TRUE = {"true", "1", "yes", "oui", "vrai", "x", "present"}
INTERNAL_COLS = {"__id", "__base_id", "_search_blob", "_score"}
SAMPLE_TEMPLATE = """first_name;last_name;email;company;function
Alice;Martin;alice.martin@example.org;Institut Imagine;Chercheuse
Hugo;Bernard;hugo.bernard@example.org;Hopital Saint-Louis;Interne
Sonia;Petit;sonia.petit@example.org;Institut Imagine;Coordinatrice
"""


def now_paris_str() -> str:
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S")


def norm_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00A0", " ").replace("\t", " ").replace("\n", " ")
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def fold_text(value: object) -> str:
    text = norm_text(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"['`´’-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_tokens(query: str) -> list[str]:
    return [token for token in fold_text(query).split(" ") if token]


def find_logo_path() -> str | None:
    base_dir = Path(__file__).resolve().parent
    for name in LOGO_CANDIDATES:
        logo_path = base_dir / name
        if logo_path.exists():
            return str(logo_path)
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    original_cols = list(df.columns)
    folded_cols = {col: fold_text(col) for col in original_cols}

    mapping: dict[str, str] = {}
    used_targets: set[str] = set()

    for standard_name, candidates in ALIASES.items():
        candidate_keys = [fold_text(candidate) for candidate in candidates]
        for col in original_cols:
            folded_col = folded_cols[col]
            if standard_name in used_targets:
                continue
            if (
                folded_col in candidate_keys
                or any(folded_col.startswith(candidate) for candidate in candidate_keys)
                or any(candidate in folded_col for candidate in candidate_keys)
            ):
                mapping[col] = standard_name
                used_targets.add(standard_name)
                break

    return df.rename(columns=mapping)


def coerce_present(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value_norm = norm_text(value)
    if value_norm in {"", "nan"}:
        return False
    return value_norm in PRESENT_TRUE


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "present" not in df.columns:
        df["present"] = False
    else:
        df["present"] = df["present"].apply(coerce_present)

    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().fillna("")
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "")
    return df


def make_base_id(row: pd.Series) -> str:
    email = norm_text(row.get("email", ""))
    if email and email != "nan":
        return f"email:{email}"
    first_name = norm_text(row.get("first_name", ""))
    last_name = norm_text(row.get("last_name", ""))
    company = norm_text(row.get("company", ""))
    return f"name:{last_name}|{first_name}|{company}"


def add_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "__base_id" not in df.columns:
        df["__base_id"] = df.apply(make_base_id, axis=1)
    df["__id"] = df["__base_id"] + "|row:" + df.index.astype(str)
    return df


def drop_internal(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(INTERNAL_COLS), errors="ignore")


def build_search_blob(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [df[col].astype(str) for col in columns if col in df.columns]
    if not parts:
        return pd.Series([""] * len(df), index=df.index)

    blob = parts[0]
    for part in parts[1:]:
        blob = blob + " " + part
    return blob.map(fold_text)


def relevance_score(row: pd.Series, query: str) -> int:
    tokens = query_tokens(query)
    if not tokens:
        return 0

    first_name = fold_text(row.get("first_name", ""))
    last_name = fold_text(row.get("last_name", ""))
    email = fold_text(row.get("email", ""))
    company = fold_text(row.get("company", ""))
    function = fold_text(row.get("function", ""))
    score = 0

    full_name = (first_name + " " + last_name).strip()
    reverse_name = (last_name + " " + first_name).strip()
    query_folded = " ".join(tokens)

    if query_folded and query_folded in {full_name, reverse_name}:
        score += 250

    for token in tokens:
        if token == email:
            score += 200
        if token == last_name:
            score += 120
        if token == first_name:
            score += 100
        if last_name.startswith(token):
            score += 90
        if first_name.startswith(token):
            score += 70
        if email.startswith(token):
            score += 60
        if company.startswith(token):
            score += 40
        if token in last_name:
            score += 50
        if token in first_name:
            score += 35
        if token in email:
            score += 30
        if token in company:
            score += 20
        if token in function:
            score += 10

    if not bool(row.get("present", False)):
        score += 5

    return score


def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    for encoding in encodings:
        try:
            return pd.read_csv(
                io.BytesIO(file_bytes),
                sep=None,
                engine="python",
                encoding=encoding,
                keep_default_na=False,
            )
        except Exception:
            continue
    raise ValueError("Le fichier CSV n'a pas pu etre lu. Verifie l'encodage ou le separateur.")


@st.cache_data(show_spinner=False)
def load_uploaded_table(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        df = read_csv_bytes(file_bytes)
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    df = standardize_columns(df)
    df = ensure_columns(df)
    df = sanitize_df(df)
    df = add_ids(df)
    return df


def state_pack(state: dict) -> str:
    raw = json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw, level=6)
    return base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")


def state_unpack(token: str) -> dict | None:
    try:
        padding = "=" * (-len(token) % 4)
        compressed = base64.urlsafe_b64decode((token + padding).encode("ascii"))
        raw = zlib.decompress(compressed)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def snapshot_present_only(df: pd.DataFrame) -> dict:
    present_df = df[df["present"]][["__base_id", "checkin_time", "checkin_by"]].copy()
    return {
        base_id: {"t": checkin_time, "b": checkin_by}
        for base_id, checkin_time, checkin_by in zip(
            present_df["__base_id"],
            present_df["checkin_time"],
            present_df["checkin_by"],
        )
    }


def apply_snapshot(df: pd.DataFrame, snapshot: dict) -> pd.DataFrame:
    df = df.copy()
    df["present"] = False
    df["checkin_time"] = ""
    df["checkin_by"] = ""

    if not snapshot:
        return df

    snapshot_df = pd.DataFrame.from_dict(snapshot, orient="index")
    snapshot_df.index.name = "__base_id"
    snapshot_df = snapshot_df.rename(columns={"t": "checkin_time", "b": "checkin_by"}).reset_index()

    df = df.merge(snapshot_df, on="__base_id", how="left", suffixes=("", "_snapshot"))
    mask = df["checkin_time_snapshot"].notna()
    df.loc[mask, "present"] = True
    df.loc[mask, "checkin_time"] = df.loc[mask, "checkin_time_snapshot"].astype(str)
    df.loc[mask, "checkin_by"] = df.loc[mask, "checkin_by_snapshot"].astype(str)
    return df.drop(columns=["checkin_time_snapshot", "checkin_by_snapshot"], errors="ignore")


def resolve_display_columns(df: pd.DataFrame) -> list[str]:
    display_cols = [col for col in DISPLAY_ORDER if col in df.columns]
    if display_cols:
        return display_cols
    return [
        col
        for col in df.columns
        if col not in {"present", "checkin_time", "checkin_by", "__id", "__base_id", "_search_blob"}
    ][:4]


def build_exports(df: pd.DataFrame) -> tuple[bytes, bytes, bytes, bytes]:
    export_df = drop_internal(df).copy()
    present_only = export_df[export_df["present"]].copy()
    absent_only = export_df[~export_df["present"]].copy()

    csv_all = export_df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    csv_present = present_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    csv_absent = absent_only.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Emargement")
        present_only.to_excel(writer, index=False, sheet_name="Presents")
        absent_only.to_excel(writer, index=False, sheet_name="Absents")
    xlsx_bytes = buffer.getvalue()

    return csv_all, csv_present, csv_absent, xlsx_bytes


def build_recent_checkins(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    recent_df = df[df["present"]].copy()
    if recent_df.empty:
        return recent_df

    recent_df["parsed_time"] = pd.to_datetime(recent_df["checkin_time"], errors="coerce")
    recent_df = recent_df.sort_values(
        by=["parsed_time", "last_name", "first_name"],
        ascending=[False, True, True],
        kind="stable",
    )
    recent_df = recent_df[["checkin_time", "first_name", "last_name", "company", "checkin_by"]].head(limit)
    return recent_df.rename(
        columns={
            "checkin_time": "Heure",
            "first_name": "Prenom",
            "last_name": "Nom",
            "company": "Societe",
            "checkin_by": "Agent",
        }
    )


def build_company_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_df = df.copy()
    summary_df["company_label"] = summary_df["company"].fillna("").astype(str).str.strip()
    summary_df.loc[summary_df["company_label"] == "", "company_label"] = "Non renseigne"

    grouped = summary_df.groupby("company_label", as_index=False).agg(
        total=("present", "size"),
        present=("present", "sum"),
    )
    grouped["absent"] = grouped["total"] - grouped["present"]
    grouped = grouped.sort_values(
        by=["present", "total", "company_label"],
        ascending=[False, False, True],
        kind="stable",
    )
    return grouped.head(12)


def reset_attendance(df: pd.DataFrame) -> pd.DataFrame:
    reset_df = df.copy()
    reset_df["present"] = False
    reset_df["checkin_time"] = ""
    reset_df["checkin_by"] = ""
    return reset_df


def qp_get(key: str, default: str = "") -> str:
    value = st.query_params.get(key, default)
    if isinstance(value, list):
        return value[0] if value else default
    return str(value)


def mailto_link(to: str, subject: str, body: str) -> str:
    params = {"subject": subject, "body": body}
    return f"mailto:{to}?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def badge_html(is_present: bool) -> str:
    if is_present:
        return "<span class='badge-present'>Present</span>"
    return "<span class='badge-todo'>A emarger</span>"


def build_brand_chips(staff_name: str, event_code: str, uploaded_name: str, total: int) -> str:
    chips: list[str] = []
    if event_code:
        chips.append(f"<span class='brand-chip'>Evenement : {escape(event_code)}</span>")
    if staff_name:
        chips.append(f"<span class='brand-chip'>Agent : {escape(staff_name)}</span>")
    if uploaded_name:
        chips.append(f"<span class='brand-chip'>Fichier : {escape(uploaded_name)}</span>")
    chips.append(f"<span class='brand-chip'>Participants : {total}</span>")
    return "".join(chips)


st.set_page_config(page_title=f"{APP_TITLE} - Institut Imagine", layout="wide")

st.markdown(
    f"""
    <style>
    :root {{
      --font-heading: "Helvetica Neue", Helvetica, Arial, sans-serif;
      --font-body: "Myriad Pro", "Helvetica Neue", Helvetica, Arial, sans-serif;
      --primary: {PRIMARY};
      --primary-dark: {PRIMARY_DARK};
      --primary-deep: {PRIMARY_DEEP};
      --primary-soft: {PRIMARY_SOFT};
      --primary-softest: {PRIMARY_SOFTEST};
      --surface: {SURFACE};
      --border: {BORDER};
      --text: {TEXT};
      --muted: {MUTED};
      --bg: {BG};
    }}

    html, body, .stApp, [class*="css"] {{
      font-family: var(--font-body) !important;
      color: var(--text);
    }}

    header[data-testid="stHeader"] {{
      display: none;
    }}

    .stApp {{
      background:
        radial-gradient(circle at top left, rgba(208, 0, 127, 0.10), transparent 28%),
        linear-gradient(180deg, #ffffff 0%, var(--bg) 52%, #f8eef4 100%);
    }}

    .block-container {{
      padding-top: 1.15rem;
      padding-bottom: 3rem;
      max-width: 1320px;
    }}

    [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #ffffff 0%, #fbf1f7 100%);
      border-right: 1px solid var(--border);
    }}

    h1, h2, h3, h4,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {{
      font-family: var(--font-heading) !important;
      color: var(--text);
    }}

    p, .stCaption, small {{
      color: var(--muted);
    }}

    [data-testid="stHorizontalBlock"] {{
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(139, 13, 89, 0.10);
      border-radius: 22px;
      padding: 0.8rem 0.95rem;
      margin-bottom: 0.65rem;
      box-shadow: 0 16px 38px rgba(101, 11, 65, 0.08);
    }}

    div[data-testid="stMetric"] {{
      background: linear-gradient(180deg, #ffffff 0%, var(--primary-softest) 100%);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 0.9rem 1rem;
      box-shadow: none;
    }}

    div[data-testid="stMetricLabel"] {{
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--muted);
      font-size: 0.74rem;
    }}

    div[data-testid="stMetricValue"] {{
      color: var(--primary-dark);
      letter-spacing: -0.02em;
    }}

    .stProgress > div > div {{
      background-color: rgba(208, 0, 127, 0.12);
    }}

    .stProgress > div > div > div > div {{
      background: linear-gradient(90deg, var(--primary-dark) 0%, var(--primary) 100%);
    }}

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stFileUploader [data-testid="stFileUploaderDropzone"] {{
      border-radius: 16px !important;
      border: 1px solid var(--border) !important;
      background: rgba(255, 255, 255, 0.94) !important;
    }}

    .stTextInput input:focus {{
      border-color: var(--primary) !important;
      box-shadow: 0 0 0 4px rgba(208, 0, 127, 0.12) !important;
    }}

    .stFileUploader [data-testid="stFileUploaderDropzone"] {{
      border-style: dashed !important;
      padding: 1rem 1.1rem;
    }}

    .stButton > button,
    .stDownloadButton > button,
    .stLinkButton > a {{
      border-radius: 999px !important;
      padding: 0.88rem 1.18rem !important;
      font-weight: 800 !important;
      min-height: 50px !important;
      white-space: nowrap !important;
      letter-spacing: 0.01em;
      transition: transform 140ms ease, box-shadow 140ms ease;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stLinkButton > a:hover {{
      transform: translateY(-1px);
    }}

    .stButton > button:not([kind="secondary"]),
    .stDownloadButton > button:not([kind="secondary"]),
    .stLinkButton > a {{
      background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%) !important;
      color: #ffffff !important;
      border: none !important;
      box-shadow: 0 12px 24px rgba(208, 0, 127, 0.18) !important;
    }}

    .stButton > button[kind="secondary"] {{
      background: rgba(255, 255, 255, 0.92) !important;
      color: var(--primary-dark) !important;
      border: 1px solid rgba(139, 13, 89, 0.24) !important;
    }}

    .badge-present {{
      background: var(--primary-soft);
      color: var(--primary-dark);
      padding: 7px 12px;
      border-radius: 999px;
      font-weight: 800;
      display: inline-block;
      white-space: nowrap;
      border: 1px solid rgba(139, 13, 89, 0.10);
    }}

    .badge-todo {{
      background: #f3edf1;
      color: var(--muted);
      padding: 7px 12px;
      border-radius: 999px;
      font-weight: 800;
      display: inline-block;
      white-space: nowrap;
      border: 1px solid rgba(113, 86, 107, 0.10);
    }}

    .cell-nowrap {{
      white-space: nowrap !important;
      overflow: hidden !important;
      text-overflow: ellipsis !important;
    }}

    .top-note {{
      background: linear-gradient(135deg, rgba(208, 0, 127, 0.08) 0%, rgba(139, 13, 89, 0.04) 100%);
      border: 1px solid rgba(208, 0, 127, 0.16);
      border-radius: 22px;
      padding: 1rem 1.1rem;
      margin-bottom: 1rem;
    }}

    .brand-hero {{
      position: relative;
      overflow: hidden;
      background: linear-gradient(135deg, rgba(208, 0, 127, 0.08) 0%, rgba(139, 13, 89, 0.04) 56%, rgba(255, 255, 255, 0.98) 100%);
      border: 1px solid rgba(208, 0, 127, 0.16);
      border-radius: 28px;
      padding: 1.3rem 1.35rem 1.2rem;
      margin: 0.2rem 0 1rem;
      box-shadow: 0 18px 44px rgba(101, 11, 65, 0.09);
    }}

    .brand-hero::after {{
      content: "";
      position: absolute;
      top: -70px;
      right: -24px;
      width: 190px;
      height: 190px;
      background: radial-gradient(circle, rgba(208, 0, 127, 0.18) 0%, rgba(208, 0, 127, 0) 72%);
      pointer-events: none;
    }}

    .brand-hero__eyebrow {{
      margin: 0 0 0.45rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 0.78rem;
      font-weight: 800;
      color: var(--primary-dark);
    }}

    .brand-hero__title {{
      margin: 0;
      font-family: var(--font-heading);
      font-size: clamp(2rem, 3vw, 3rem);
      line-height: 0.95;
      letter-spacing: -0.03em;
      color: var(--primary-dark);
      font-weight: 700;
    }}

    .brand-hero__subtitle {{
      margin: 0.55rem 0 0;
      max-width: 760px;
      color: var(--text);
      font-size: 1.02rem;
    }}

    .brand-hero__signature {{
      margin: 0.85rem 0 0;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      font-size: 0.88rem;
      font-weight: 800;
      color: var(--primary-dark);
    }}

    .brand-chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.58rem;
      margin-top: 1rem;
    }}

    .brand-chip {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.42rem 0.78rem;
      border-radius: 999px;
      border: 1px solid rgba(139, 13, 89, 0.14);
      background: rgba(255, 255, 255, 0.88);
      color: var(--primary-dark);
      font-size: 0.82rem;
      font-weight: 700;
    }}

    [data-testid="stDataFrameResizable"] {{
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid var(--border);
    }}

    [data-testid="stAlert"] {{
      border-radius: 18px;
      border: 1px solid rgba(139, 13, 89, 0.10);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if "_run_count" not in st.session_state:
    st.session_state._run_count = 0
st.session_state._run_count += 1

logo_path = find_logo_path()
logo_col, title_col = st.columns([1.15, 6])
with logo_col:
    if logo_path:
        st.image(logo_path, width=112)
with title_col:
    st.markdown(f"## {APP_TITLE}")
    st.caption("Institut Imagine")
st.divider()

with st.sidebar:
    st.header("Demarrage")
    staff_name = st.text_input("Nom de l'agent", placeholder="Doralis").strip()
    event_code = st.text_input("Code evenement", placeholder="Seminaire 2026").strip()
    tablet_mode = st.toggle("Mode tablette", value=True)
    st.download_button(
        "Telecharger un modele CSV",
        data=SAMPLE_TEMPLATE.encode("utf-8-sig"),
        file_name="modele_emargement.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("Colonnes reconnues : first_name, last_name, email, company, function")

uploaded = st.file_uploader("Importer un fichier participants (.xlsx ou .csv)", type=["xlsx", "csv"])
if uploaded is None:
    st.markdown(
        """
        <div class="top-note">
          Charge un fichier participants pour demarrer
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = hashlib.sha256((uploaded.name + "::").encode("utf-8") + file_bytes).hexdigest()[:16]
resume_warning = ""

new_file = ("file_hash" not in st.session_state) or (st.session_state.get("file_hash") != file_hash)
if new_file:
    st.session_state.file_hash = file_hash
    st.session_state.filename = uploaded.name
    st.session_state.page = 1
    st.session_state["_prev_query"] = ""

    try:
        df = load_uploaded_table(file_bytes, uploaded.name).copy()
    except Exception as exc:
        st.error(f"Impossible de lire le fichier : {exc}")
        st.stop()

    resume_token = qp_get("r", "")
    if resume_token:
        payload = state_unpack(resume_token)
        if payload and payload.get("h") == file_hash:
            snapshot = payload.get("s", {})
            df = apply_snapshot(df, snapshot)
            st.session_state.snap = snapshot
        else:
            st.session_state.snap = snapshot_present_only(df)
            resume_warning = "Le lien de reprise ne correspond pas a ce fichier, la session a ete redemarree."
    else:
        st.session_state.snap = snapshot_present_only(df)

    display_cols = resolve_display_columns(df)
    search_cols = list(dict.fromkeys(display_cols + [col for col in ["email", "company", "function"] if col in df.columns]))
    df["_search_blob"] = build_search_blob(df, search_cols)

    st.session_state.df = df
    st.session_state.id2i = dict(zip(df["__id"], df.index))
    st.session_state.dirty_qp = True
    st.session_state._last_packed = None

df = st.session_state.df

with st.sidebar:
    st.divider()
    st.subheader("Session")
    st.caption(f"Fichier charge : {uploaded.name}")
    st.caption(f"Participants detectes : {len(df)}")
    reset_guard = st.checkbox("Autoriser la reinitialisation", key="reset_guard")
    if st.button(
        "Reinitialiser tous les emargements",
        type="secondary",
        disabled=not reset_guard,
        use_container_width=True,
    ):
        st.session_state.df = reset_attendance(df)
        st.session_state.snap = {}
        st.session_state.dirty_qp = True
        st.rerun()

if resume_warning:
    st.warning(resume_warning)

if not staff_name:
    st.warning("Saisis le nom de l'agent pour activer les boutons Emarger et Annuler.")

st.markdown(
    f"""
    <section class="brand-hero">
      <p class="brand-hero__eyebrow">Institut Imagine</p>
      <h1 class="brand-hero__title">{APP_TITLE}</h1>
      <p class="brand-hero__subtitle">{APP_SUBTITLE}</p>
      <p class="brand-hero__signature">{APP_TAGLINE}</p>
      <div class="brand-chip-row">{build_brand_chips(staff_name, event_code, uploaded.name, len(df))}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

total = len(df)
present_count = int(df["present"].sum())
remaining = total - present_count
attendance_rate = round((present_count / total) * 100) if total else 0

metric_cols = st.columns([1, 1, 1, 1.2])
metric_cols[0].metric("Participants", total)
metric_cols[1].metric("Presents", present_count)
metric_cols[2].metric("Restants", remaining)
metric_cols[3].metric("Taux de presence", f"{attendance_rate}%")
st.progress((present_count / total) if total else 0.0, text=f"Progression de l'emargement : {attendance_rate}%")

search_col, company_col = st.columns([2.2, 1.2])
with search_col:
    query = st.text_input(
        "Recherche",
        placeholder="Nom, prenom, email, societe...",
        key="search_query",
    ).strip()

companies = sorted({value.strip() for value in df["company"].astype(str).tolist() if value.strip()})
company_options = ["Toutes les societes"] + companies
with company_col:
    company_filter = st.selectbox("Societe", company_options, index=0)

prev_query = st.session_state.get("_prev_query", "")
if query != prev_query:
    st.session_state.page = 1
st.session_state["_prev_query"] = query

filter_choice = st.radio("Filtre", ["Non emarges", "Tous", "Presents uniquement"], index=0, horizontal=True)
st.divider()

view = df.copy()
if query:
    tokens = query_tokens(query)
    mask = pd.Series(True, index=view.index)
    for token in tokens:
        mask &= view["_search_blob"].str.contains(re.escape(token), na=False, regex=True)
    view = view[mask].copy()
    view["_score"] = view.apply(lambda row: relevance_score(row, query), axis=1)
    view = view.sort_values(by=["_score"], ascending=False, kind="stable")
else:
    view = view.sort_values(by=["last_name", "first_name"], ascending=[True, True], kind="stable")

if company_filter != "Toutes les societes":
    view = view[view["company"].astype(str).str.strip() == company_filter].copy()

if filter_choice == "Presents uniquement":
    view = view[view["present"]].copy()
elif filter_choice == "Non emarges":
    view = view[~view["present"]].copy()

page_size_options = [25, 50, 75, 100]
default_page_size = 25 if tablet_mode else 50
page_size = st.selectbox(
    "Participants par page",
    page_size_options,
    index=page_size_options.index(default_page_size),
    key="page_size",
)

total_rows = len(view)
page_count = max(1, (total_rows + page_size - 1) // page_size)
if "page" not in st.session_state:
    st.session_state.page = 1
st.session_state.page = min(max(1, st.session_state.page), page_count)


def pager(current_page_count: int, page_value: int, label: str) -> None:
    prev_col, info_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        if st.button(
            "Page precedente",
            disabled=(page_value <= 1),
            key=f"prev_{label}",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.page = max(1, page_value - 1)
            st.rerun()
    with info_col:
        st.markdown(
            f"<div style='text-align:center; font-weight:800; padding:0.35rem 0; color:{PRIMARY_DARK};'>Page {page_value} / {current_page_count}</div>",
            unsafe_allow_html=True,
        )
    with next_col:
        if st.button(
            "Page suivante",
            disabled=(page_value >= current_page_count),
            key=f"next_{label}",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.page = min(current_page_count, page_value + 1)
            st.rerun()


pager(page_count, st.session_state.page, label="top")
start = (st.session_state.page - 1) * page_size
end = start + page_size
view_page = view.iloc[start:end].copy()

st.subheader("Liste des participants")
header = st.columns([2.2, 2.4, 3.3, 2.8, 1.8, 2.0, 2.1])
header[0].markdown("**Prenom**")
header[1].markdown("**Nom**")
header[2].markdown("**Email**")
header[3].markdown("**Societe**")
header[4].markdown("**Heure**")
header[5].markdown("**Statut**")
header[6].markdown("**Action**")

can_edit = bool(staff_name)
for _, row in view_page.iterrows():
    row_id = row["__id"]
    is_present = bool(row["present"])
    checkin_time = str(row.get("checkin_time", "")).strip()
    hour_display = checkin_time[-8:] if len(checkin_time) >= 8 else checkin_time

    row_cols = st.columns([2.2, 2.4, 3.3, 2.8, 1.8, 2.0, 2.1])
    row_cols[0].markdown(f"<div class='cell-nowrap'>{escape(str(row.get('first_name', '')))}</div>", unsafe_allow_html=True)
    row_cols[1].markdown(f"<div class='cell-nowrap'>{escape(str(row.get('last_name', '')))}</div>", unsafe_allow_html=True)
    row_cols[2].markdown(f"<div class='cell-nowrap'>{escape(str(row.get('email', '')))}</div>", unsafe_allow_html=True)
    row_cols[3].markdown(f"<div class='cell-nowrap'>{escape(str(row.get('company', '')))}</div>", unsafe_allow_html=True)
    row_cols[4].markdown(f"<div class='cell-nowrap'>{escape(hour_display)}</div>", unsafe_allow_html=True)
    row_cols[5].markdown(badge_html(is_present), unsafe_allow_html=True)

    row_index = st.session_state.id2i.get(row_id)
    if not is_present:
        if row_cols[6].button(
            "Emarger",
            key=f"checkin_{row_id}",
            use_container_width=True,
            disabled=not can_edit,
        ):
            if row_index is not None:
                df.at[row_index, "present"] = True
                df.at[row_index, "checkin_time"] = now_paris_str()
                df.at[row_index, "checkin_by"] = staff_name
                st.session_state.df = df

                base_id = df.at[row_index, "__base_id"]
                snapshot = st.session_state.get("snap", {})
                snapshot[base_id] = {
                    "t": df.at[row_index, "checkin_time"],
                    "b": df.at[row_index, "checkin_by"],
                }
                st.session_state.snap = snapshot
                st.session_state.dirty_qp = True
            st.rerun()
    else:
        if row_cols[6].button(
            "Annuler",
            key=f"cancel_{row_id}",
            use_container_width=True,
            type="secondary",
            disabled=not can_edit,
        ):
            if row_index is not None:
                df.at[row_index, "present"] = False
                df.at[row_index, "checkin_time"] = ""
                df.at[row_index, "checkin_by"] = ""
                st.session_state.df = df

                base_id = df.at[row_index, "__base_id"]
                snapshot = st.session_state.get("snap", {})
                snapshot.pop(base_id, None)
                st.session_state.snap = snapshot
                st.session_state.dirty_qp = True
            st.rerun()

pager(page_count, st.session_state.page, label="bottom")
if total_rows == 0:
    st.caption("Affichage : 0 / 0")
else:
    st.caption(f"Affichage : {start + 1}-{min(end, total_rows)} / {total_rows}")

st.divider()
summary_left, summary_right = st.columns([1.1, 1])
with summary_left:
    st.subheader("Derniers emargements")
    recent_checkins = build_recent_checkins(df)
    if recent_checkins.empty:
        st.info("Aucun participant n'a encore ete emarge.")
    else:
        st.dataframe(recent_checkins, use_container_width=True, hide_index=True)

with summary_right:
    st.subheader("Synthese par societe")
    company_summary = build_company_summary(df)
    if company_summary.empty:
        st.info("Aucune societe exploitable dans ce fichier.")
    else:
        chart_source = company_summary.melt(
            id_vars=["company_label", "total"],
            value_vars=["present", "absent"],
            var_name="status",
            value_name="count",
        )
        chart_source["status"] = chart_source["status"].map({"present": "Presents", "absent": "Absents"})
        chart = (
            alt.Chart(chart_source)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(
                y=alt.Y("company_label:N", sort="-x", title="Societe"),
                x=alt.X("count:Q", stack="zero", title="Participants"),
                color=alt.Color(
                    "status:N",
                    title="Statut",
                    scale=alt.Scale(domain=["Presents", "Absents"], range=[PRIMARY_DARK, "#E7D4E0"]),
                ),
                tooltip=[
                    alt.Tooltip("company_label:N", title="Societe"),
                    alt.Tooltip("status:N", title="Statut"),
                    alt.Tooltip("count:Q", title="Volume"),
                    alt.Tooltip("total:Q", title="Total"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

st.divider()
if st.session_state.get("dirty_qp", False) and st.session_state._run_count >= 2:
    packed_state = state_pack({"h": file_hash, "s": st.session_state.get("snap", {})})
    if st.session_state.get("_last_packed") != packed_state:
        st.session_state._last_packed = packed_state
        st.query_params["r"] = packed_state
        st.query_params["v"] = APP_BUILD
    st.session_state.dirty_qp = False

st.subheader("Exports")
csv_all, csv_present, csv_absent, xlsx_bytes = build_exports(df)
export_cols = st.columns([1, 1, 1, 1])
with export_cols[0]:
    st.download_button("CSV complet", data=csv_all, file_name="emargement_complet.csv", mime="text/csv", use_container_width=True)
with export_cols[1]:
    st.download_button("CSV presents", data=csv_present, file_name="emargement_presents.csv", mime="text/csv", use_container_width=True)
with export_cols[2]:
    st.download_button("CSV absents", data=csv_absent, file_name="emargement_absents.csv", mime="text/csv", use_container_width=True)
with export_cols[3]:
    st.download_button(
        "Excel (.xlsx)",
        data=xlsx_bytes,
        file_name="emargement_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.caption("Astuce : pour reprendre la session sur une autre tablette, copie simplement l'URL du navigateur.")
st.divider()

st.subheader("Envoyer les exports par email")
timestamp = datetime.now(PARIS_TZ).strftime("%Y-%m-%d_%H%M")
email_subject = f"[Emargement] {event_code or Path(uploaded.name).stem} - presents / absents - {timestamp}"
email_body = (
    "Bonjour,\n\n"
    "Veuillez trouver en pieces jointes :\n"
    "- la liste des presents\n"
    "- la liste des absents\n\n"
    "Pieces jointes a ajouter :\n"
    "1) emargement_presents.csv\n"
    "2) emargement_absents.csv\n\n"
    f"Agent : {staff_name or 'non renseigne'}\n"
    f"Fichier : {st.session_state.get('filename', '')}\n"
    f"Horodatage : {now_paris_str()} (Europe/Paris)\n"
)

st.caption("Le bouton ouvre ton application mail. Les pieces jointes restent a ajouter manuellement.")
st.link_button("Ouvrir l'application mail", url=mailto_link(MAIL_TO, email_subject, email_body), use_container_width=True)
