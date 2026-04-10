import builtins
import re

import streamlit as st


# Fallback so removing APP_TAGLINE from app.py does not crash startup.
builtins.APP_TAGLINE = ""

_original_markdown = st.markdown
_signature_pattern = re.compile(
    r"<p\s+class=\"brand-hero__signature\">.*?</p>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _patched_markdown(body, *args, **kwargs):
    if isinstance(body, str) and "brand-hero__signature" in body:
        body = _signature_pattern.sub("", body)
    return _original_markdown(body, *args, **kwargs)


st.markdown = _patched_markdown
