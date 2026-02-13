# app.py
# Table d’émargement — Institut Imagine (Streamlit)
# - Recherche en live (sans appuyer sur Entrée) via streamlit-keyup
# - Import Excel (.xlsx) avec mapping intelligent des colonnes (FR/EN)
# - 1 ligne = 1 participant, pagination en bas
# - Horodatage en heure locale FR (Europe/Paris)
# - Export CSV propre
# - UI optimisée tablette (boutons/labels plus compacts)

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit_keyup import st_keyup

# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "Outil d’émargement — Institut Imagine"
TZ = ZoneInfo("Europe/Paris")  # corrige le décalage d’1h
PRIMARY = "#c1007a"  # rose Imagine (approx)
BG = "#f6f7fb"

# Logo: le fichier doit exister dans le repo à la racine (ou adapte le chemin)
LOGO_PATH = "LOGO ROSE.png"

# Pagination
DEFAULT_PAGE_SIZE = 25
PAGE_SIZES = [10, 25, 50, 100]

# Colonnes internes standardisées
STD_COLS = {
    "first_name": "Prénom",
    "last_name": "Nom",
    "email": "Email",
    "company": "Société",
    "role": "Fonction",
    "present": "Présent",
    "checkin_time": "Heure d’émargement",
    "checkin_by": "Émargé par",
}

# -----------------------------
# UI / CSS
# -----------------------------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          html, body, [class*="css"] {{
            background: {BG};
          }}

          /* Hide Streamlit default menu/footer */
          #MainMenu {{visibility: hidden;}}
          footer {{visibility: hidden;}}
          header {{visibility: hidden;}}

          /* Header card */
          .im-header {{
            background: white;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            display: flex;
            align-items: center;
            gap: 18px;
            margin-bottom: 18px;
          }}
          .im-title {{
            font-size: 30px;
            font-weight: 800;
            margin: 0;
            line-height: 1.1;
            color: #111827;
          }}
          .im-sub {{
            margin-top: 6px;
            color: #6b7280;
            font-size: 14px;
          }}

          /* Metric cards */
          .metric-row {{
            background: white;
            border-radius: 16px;
            padding: 10px 14px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
          }}
          .metric-label {{
            color:#6b7280;
            font-size: 13px;
          }}
          .metric-value {{
            font-size: 34px;
            font-weight: 800;
            color: #111827;
            line-height: 1.0;
          }}

          /* Participant rows */
          .row-card {{
            background: white;
            border-radius: 16px;
            padding: 10px 12px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 10px;
          }}

          .pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 13px;
            white-space: nowrap;
          }}
          .pill-todo {{
            background: #f3f4f6;
            color: #374151;
          }}
          .pill-ok {{
            background: #e7f7ee;
            color: #0f6b3d;
          }}

          /* Buttons */
          div.stButton > button {{
            border-radius: 14px !important;
            border: 2px solid {PRIMARY} !important;
            font-weight: 800 !important;
            padding: 0.55rem 0.9rem !important;
            width: 100% !important;
          }}
          /* Primary button style: we’ll apply via st.button + class hack using :has is not reliable,
             so we use consistent style and rely on label / context. */
          .btn-primary div.stButton > button {{
            background: {PRIMARY} !important;
            color: white !important; /* important: texte blanc */
          }}
          .btn-outline div.stButton > button {{
            background: white !important;
            color: {PRIMARY} !important;
          }}

          /* Table header mimic */
          .table-head {{
            display: grid;
            grid-template-columns: 1.2fr 1.2fr 1.8fr 1.4fr 1.2fr 1fr 1.1fr;
            gap: 10px;
            color: #6b7280;
            font-weight: 800;
            font-size: 14px;
            padding: 0 6px 6px 6px;
          }}
          .table-row {{
            display: grid;
            grid-template-columns: 1.2fr 1.2fr 1.8fr 1.4fr 1.2fr 1fr 1.1fr;
            gap: 10px;
            align-items: center;
          }}

          /* Keep last name on one line (no wrap), but don’t truncate on desktop.
             On tablet, we reduce font-size slightly + keep nowrap (full name must stay on same line). */
          .nowrap {{
            white-space: nowrap;
            overflow: visible;
            text-overflow: clip;
          }}

          /* Tablet tweaks */
          @media (max-width: 900px) {{
            .im-title {{ font-size: 24px; }}
            .metric-value {{ font-size: 28px; }}
            .pill {{ font-size: 12px; padding: 5px 8px; }}
            div.stButton > button {{ padding: 0.5rem 0.7rem !important; font-size: 13px !important; }}
            .table-head, .table-row {{
              grid-template-columns: 1.3fr 1.4fr 1.6fr 1.3fr 1.0fr 0.9fr 1.0fr;
            }}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# HELPERS: column mapping
# -----------------------------
def normalize_header(h: str) -> str:
    h = (h or "").strip()
    h = h.replace("\n", " ").replace("\t", " ")
    h = re.sub(r"\s+", " ", h)
    return h.lower()

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # match by normalized header
    norm_map = {normalize_header(c): c for c in df.columns}
    cand_norm = [normalize_header(c) for c in candidates]
    for cn in cand_norm:
        if cn in norm_map:
            return norm_map[cn]
    # also allow fuzzy contains (e.g. "First name (required)")
    for col in df.columns:
        n = normalize_header(col)
        for cn in cand_norm:
            if cn and cn in n:
                return col
    return None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # candidates for each standardized field
    cands = {
        "first_name": ["prénom", "prenom", "first name", "firstname", "given name", "forename"],
        "last_name": ["nom", "last name", "lastname", "surname", "family name"],
        "email": ["email", "e-mail", "mail", "courriel"],
        "company": ["société", "societe", "company", "organisation", "organization", "structure", "institution"],
        "role": ["fonction", "poste", "job title", "title", "role", "position"],
        "present": ["présent", "present", "emarge", "émargé", "émargement", "checked in", "checkin", "check-in"],
        "checkin_time": ["heure", "heure d’émargement", "heure d'emargement", "checkin_time", "check-in time", "checkin time"],
        "checkin_by": ["émargé par", "emargé par", "emarge par", "checkin_by", "checked in by", "agent"],
    }

    mapping: Dict[str, Optional[str]] = {k: find_col(df, v) for k, v in cands.items()}

    out = pd.DataFrame()

    # Always keep any original columns? Non. We normalize to the required set.
    out["first_name"] = df[mapping["first_name"]] if mapping["first_name"] else ""
    out["last_name"] = df[mapping["last_name"]] if mapping["last_name"] else ""
    out["email"] = df[mapping["email"]] if mapping["email"] else ""
    out["company"] = df[mapping["company"]] if mapping["company"] else ""
    out["role"] = df[mapping["role"]] if mapping["role"] else ""

    # present / time / by
    if mapping["present"]:
        # accept True/False, 1/0, "yes"/"no", etc.
        col = df[mapping["present"]]
        out["present"] = col.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "oui", "present", "présent"])
    else:
        out["present"] = False

    out["checkin_time"] = df[mapping["checkin_time"]] if mapping["checkin_time"] else ""
    out["checkin_by"] = df[mapping["checkin_by"]] if mapping["checkin_by"] else ""

    # Clean
    for c in ["first_name", "last_name", "email", "company", "role", "checkin_time", "checkin_by"]:
        out[c] = out[c].fillna("").astype(str).str.strip()

    # Build search index
    out["_search"] = (
        out["first_name"].fillna("") + " " +
        out["last_name"].fillna("") + " " +
        out["email"].fillna("") + " " +
        out["company"].fillna("") + " " +
        out["role"].fillna("")
    ).str.lower()

    # Stable id to avoid duplicate widget keys (index can change with filters)
    out["_rid"] = (
        out["email"].str.lower().replace("", pd.NA)
        .fillna(out["first_name"].str.lower() + "|" + out["last_name"].str.lower() + "|" + out["company"].str.lower())
    )
    # if still duplicates, add row number suffix
    out["_rid"] = out["_rid"].astype(str)
    out["_rid"] = out["_rid"] + "|" + pd.Series(range(len(out))).astype(str)

    return out


# -----------------------------
# STATE
# -----------------------------
@dataclass
class AppState:
    df: Optional[pd.DataFrame] = None
    filename: str = ""

def get_state() -> AppState:
    if "state" not in st.session_state:
        st.session_state["state"] = AppState()
    return st.session_state["state"]


# -----------------------------
# EXPORT
# -----------------------------
def export_csv_bytes(df: pd.DataFrame) -> bytes:
    # Export only useful columns (standard), in a friendly order
    exp = df.copy()
    exp = exp[["first_name", "last_name", "email", "company", "role", "present", "checkin_time", "checkin_by"]]
    csv = exp.to_csv(index=False)
    return csv.encode("utf-8-sig")  # Excel-friendly


# -----------------------------
# CHECKIN ACTIONS
# -----------------------------
def now_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def do_checkin(df: pd.DataFrame, rid: str, agent: str) -> pd.DataFrame:
    # set present True + time + agent for row rid
    mask = df["_rid"] == rid
    if not mask.any():
        return df
    df.loc[mask, "present"] = True
    df.loc[mask, "checkin_time"] = now_str()
    df.loc[mask, "checkin_by"] = agent.strip()
    return df

def undo_checkin(df: pd.DataFrame, rid: str) -> pd.DataFrame:
    mask = df["_rid"] == rid
    if not mask.any():
        return df
    df.loc[mask, "present"] = False
    df.loc[mask, "checkin_time"] = ""
    df.loc[mask, "checkin_by"] = ""
    return df


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()

    state = get_state()

    # Sidebar settings
    with st.sidebar:
        st.markdown("## Réglages")
        agent = st.text_input("Nom de l'agent (optionnel)", placeholder="Ex: Ambroise", key="agent_name")
        st.caption("Ce nom sera enregistré dans la colonne *checkin_by*.")
        st.divider()
        st.caption("Astuce : l'export CSV est compatible Excel (FR).")

    # Header
    col_logo, col_text = st.columns([1, 6], vertical_alignment="center")
    with col_logo:
        try:
            st.image(LOGO_PATH, width=180)  # logo plus petit
        except Exception:
            st.write("")  # pas bloquant

    with col_text:
        st.markdown(f"<div class='im-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='im-sub'>Importez votre liste, recherchez un participant, émargez, puis exportez la feuille d’émargement.</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    # Upload
    up = st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False)

    if up is not None:
        try:
            raw = pd.read_excel(up)
            df = standardize_columns(raw)
            state.df = df
            state.filename = up.name
            st.toast("Liste importée ✅", icon="✅")
        except Exception as e:
            st.error("Impossible de lire ce fichier Excel. Vérifie qu'il s'agit bien d'un .xlsx valide.")
            st.stop()

    if state.df is None:
        st.info("📩 Importez un fichier Excel pour commencer.")
        return

    df = state.df

    # Pie chart (camembert) au-dessus du bandeau
    total = int(len(df))
    present_n = int(df["present"].sum())
    remaining_n = total - present_n

    # Simple chart using st.pyplot would require matplotlib, but Streamlit has built-in charting via altair in st.altair_chart.
    # To avoid extra deps, use st.bar_chart? But they asked camembert; Streamlit supports altair by default.
    import altair as alt  # available with Streamlit

    pie_df = pd.DataFrame(
        {"Statut": ["Présents", "Restants"], "Nombre": [present_n, remaining_n]}
    )
    pie = (
        alt.Chart(pie_df)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta(field="Nombre", type="quantitative"),
            color=alt.Color(field="Statut", type="nominal", legend=alt.Legend(title="")),
            tooltip=["Statut", "Nombre"],
        )
        .properties(height=220)
    )
    st.altair_chart(pie, use_container_width=True)

    # Metric band + search (live)
    m1, m2, m3, m4 = st.columns([1, 1, 1, 3], vertical_alignment="center")
    with m1:
        st.markdown("<div class='metric-row'><div class='metric-label'>Participants</div>"
                    f"<div class='metric-value'>{total}</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-row'><div class='metric-label'>Présents</div>"
                    f"<div class='metric-value'>{present_n}</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-row'><div class='metric-label'>Restants</div>"
                    f"<div class='metric-value'>{remaining_n}</div></div>", unsafe_allow_html=True)

    with m4:
        # Recherche live: pas besoin d'Entrée
        search = st_keyup(
            "Recherche",
            placeholder="Nom, prénom, email, société…",
            key="search_keyup",
        )
        search = (search or "").strip().lower()

    # Filters row
    f1, f2, f3 = st.columns([1.2, 1.2, 3.6])
    with f1:
        only_not = st.checkbox("Non émargés", value=False, key="only_not")
    with f2:
        only_present = st.checkbox("Présents uniquement", value=False, key="only_present")
    with f3:
        st.caption("La recherche remonte automatiquement les meilleurs résultats.")

    # Apply filters
    view = df.copy()

    if only_not and only_present:
        # mutually exclusive – keep "présents" as priority
        only_not = False

    if only_not:
        view = view[view["present"] == False]
    if only_present:
        view = view[view["present"] == True]

    if search:
        view = view[view["_search"].str.contains(re.escape(search), na=False)]

    # Sort: best matches first when searching
    if search:
        # simple ranking: startswith > contains
        def rank_row(s: str) -> int:
            if not isinstance(s, str):
                return 2
            if s.startswith(search):
                return 0
            if f" {search}" in s:
                return 0
            if search in s:
                return 1
            return 2
        view = view.assign(_rank=view["_search"].map(rank_row)).sort_values(["_rank", "last_name", "first_name"])
    else:
        view = view.sort_values(["last_name", "first_name"])

    # Export
    st.write("")
    exp_col1, exp_col2, exp_col3 = st.columns([2, 2, 6], vertical_alignment="center")
    with exp_col1:
        page_size = st.selectbox("Taille de page", PAGE_SIZES, index=PAGE_SIZES.index(DEFAULT_PAGE_SIZE), key="page_size")
    with exp_col2:
        csv_bytes = export_csv_bytes(df)
        st.download_button(
            "⬇️ Exporter la feuille (CSV)",
            data=csv_bytes,
            file_name=f"emargement_export_{datetime.now(TZ).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col3:
        st.write("")

    st.markdown("## Liste des participants")

    # Pagination (controls BELOW filters, as requested)
    total_view = int(len(view))
    pages = max(1, math.ceil(total_view / int(page_size)))
    if "page" not in st.session_state:
        st.session_state["page"] = 1

    # clamp page
    st.session_state["page"] = max(1, min(st.session_state["page"], pages))

    p_left, p_mid, p_right = st.columns([2, 2, 2], vertical_alignment="center")
    with p_left:
        st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
        if st.button("⬅️  Page précédente", use_container_width=True, key="prev_page"):
            st.session_state["page"] = max(1, st.session_state["page"] - 1)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with p_mid:
        st.markdown(f"<div style='text-align:center; font-weight:800;'>Page {st.session_state['page']} / {pages}</div>", unsafe_allow_html=True)

    with p_right:
        st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
        if st.button("Page suivante  ➡️", use_container_width=True, key="next_page"):
            st.session_state["page"] = min(pages, st.session_state["page"] + 1)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Page slice
    start = (st.session_state["page"] - 1) * int(page_size)
    end = start + int(page_size)
    page_df = view.iloc[start:end].copy()

    # Header row
    st.markdown(
        """
        <div class="row-card" style="padding: 12px 12px 6px 12px;">
          <div class="table-head">
            <div>Prénom</div>
            <div>Nom</div>
            <div>Email</div>
            <div>Société</div>
            <div>Fonction</div>
            <div>Statut</div>
            <div style="text-align:right;">Action</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Participant rows
    for _, r in page_df.iterrows():
        rid = r["_rid"]
        is_present = bool(r["present"])

        # card container
        st.markdown("<div class='row-card'>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6, c7 = st.columns([1.2, 1.2, 1.8, 1.4, 1.2, 1.0, 1.1], vertical_alignment="center")

        with c1:
            st.markdown(f"<div class='nowrap'>{r['first_name']}</div>", unsafe_allow_html=True)
        with c2:
            # ensure last name full + single line
            st.markdown(f"<div class='nowrap' style='font-weight:800;'>{r['last_name']}</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='nowrap'>{r['email']}</div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='nowrap'>{r['company']}</div>", unsafe_allow_html=True)
        with c5:
            st.markdown(f"<div class='nowrap'>{r['role']}</div>", unsafe_allow_html=True)
        with c6:
            if is_present:
                st.markdown("<span class='pill pill-ok'>✅ Présent</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='pill pill-todo'>🟪 À émarger</span>", unsafe_allow_html=True)

        with c7:
            if is_present:
                st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
                if st.button("Annuler", key=f"undo_{rid}", use_container_width=True):
                    state.df = undo_checkin(state.df, rid)
                    st.toast("Émargement annulé", icon="↩️")
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
                if st.button("Émarger", key=f"check_{rid}", use_container_width=True):
                    state.df = do_checkin(state.df, rid, agent or "")
                    st.toast("Émargé ✅", icon="✅")
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Pagination again at bottom (optional but useful on long pages)
    st.write("")
    p2_left, p2_mid, p2_right = st.columns([2, 2, 2], vertical_alignment="center")
    with p2_left:
        st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
        if st.button("⬅️  Page précédente", use_container_width=True, key="prev_page_bottom"):
            st.session_state["page"] = max(1, st.session_state["page"] - 1)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with p2_mid:
        st.markdown(f"<div style='text-align:center; font-weight:800;'>Page {st.session_state['page']} / {pages}</div>", unsafe_allow_html=True)
    with p2_right:
        st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
        if st.button("Page suivante  ➡️", use_container_width=True, key="next_page_bottom"):
            st.session_state["page"] = min(pages, st.session_state["page"] + 1)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
