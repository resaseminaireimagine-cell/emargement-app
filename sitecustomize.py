import builtins
import re

import streamlit as st


# Fallback so removing APP_TAGLINE from app.py does not crash startup.
builtins.APP_TAGLINE = ""

_FONT_SNIPPET = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stMarkdownContainer"], [data-testid="stText"], label, input, textarea, select, button, p, span, div, h1, h2, h3, h4, h5, h6 {
    font-family: "Montserrat", sans-serif !important;
}
</style>
"""

_original_markdown = st.markdown
_signature_pattern = re.compile(
    r"<p\s+class=\"brand-hero__signature\">.*?</p>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _patched_markdown(body, *args, **kwargs):
    if isinstance(body, str):
        if "brand-hero__signature" in body:
            body = _signature_pattern.sub("", body)
        if not getattr(builtins, "_codex_montserrat_injected", False):
            body = _FONT_SNIPPET + body
            builtins._codex_montserrat_injected = True
    return _original_markdown(body, *args, **kwargs)


st.markdown = _patched_markdown
