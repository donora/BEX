"""BEX's colours: EcoLabel's palettes, so the two read as one family.

Evening (dark, the default) and Meadow (light; `BEX_THEME=meadow`) are the
EcoLabel app's palettes of the same names (ecolabel/app/src/theme.tsx). Streamlit
takes them as `--theme.*` flags (`bex app` passes them: Streamlit only reads
.streamlit/config.toml from the folder it's started in, which mirrors this);
pages take the highlight colours from `colours()`.
"""
from __future__ import annotations

import os

PALETTES = {
    "evening": {
        "dark": True,
        "bg": "#16211C", "card": "#1F2C26", "ink": "#ECEADD", "muted": "#A3AE9F", "line": "#33433A",
        "primary": "#8CC79A", "primary_deep": "#A6D6B1", "primary_soft": "#2A3D33", "on_primary": "#10201A",
        "accent": "#D8BB72", "glow": "#9CF0B0", "night": "#0B1310",
        "practice": "#F4A7BD", "yes": "#6FAF7E", "no": "#C98468", "maybe": "#C9AA62",
        "researcher": "#8C78E0",
    },
    "meadow": {
        "dark": False,
        "bg": "#F3F1E7", "card": "#FBFAF4", "ink": "#24342B", "muted": "#6E7A6F", "line": "#E2DECF",
        "primary": "#4E7D5B", "primary_deep": "#3E6649", "primary_soft": "#DDE9DC", "on_primary": "#FFFFFF",
        "accent": "#C7A659", "glow": "#6CCB86", "night": "#13221D",
        "practice": "#E58BA5", "yes": "#4E8A5F", "no": "#B26B52", "maybe": "#C2A15A",
        "researcher": "#7B63D1",
    },
}
DEFAULT = "evening"


def name() -> str:
    n = os.environ.get("BEX_THEME", DEFAULT).lower()
    return n if n in PALETTES else DEFAULT


def colours() -> dict:
    return PALETTES[name()]


def streamlit_options(n: str | None = None) -> dict[str, str]:
    c = PALETTES[n or name()]
    return {
        "base": "dark" if c["dark"] else "light",
        "primaryColor": c["primary"], "backgroundColor": c["bg"], "secondaryBackgroundColor": c["card"],
        "textColor": c["ink"], "borderColor": c["line"], "dataframeBorderColor": c["line"],
        "dataframeHeaderBackgroundColor": c["card"], "linkColor": c["primary"],
        "codeBackgroundColor": c["night"], "greenColor": c["yes"], "redColor": c["no"],
        "yellowColor": c["accent"], "orangeColor": c["maybe"], "violetColor": c["researcher"],
        "grayColor": c["muted"], "baseRadius": "10px",
    }


def flags(n: str | None = None) -> list[str]:
    return [f"--theme.{k}={v}" for k, v in streamlit_options(n).items()]


def config_toml(n: str | None = None) -> str:
    lines = ["# Generated from bex/theme.py (EcoLabel's palettes): edit there, then",
             "#   .venv/bin/python -c 'from bex import theme; print(theme.config_toml(), end=\"\")' > .streamlit/config.toml",
             "[theme]"]
    lines += [f'{k} = "{v}"' for k, v in streamlit_options(n).items()]
    return "\n".join(lines) + "\n"


def css(n: str | None = None) -> str:
    """What the theme options can't reach: text on primary buttons (Evening's
    light green needs dark text) and the app's highlight colours as CSS
    variables for pages' own HTML."""
    c = PALETTES[n or name()]
    vars_ = "".join(f"--bex-{k.replace('_', '-')}:{v};" for k, v in c.items() if isinstance(v, str))
    return ("<style>:root{" + vars_ + "}"
            "[data-testid='stBaseButton-primary'],[data-testid='stBaseButton-primaryFormSubmit']"
            f"{{color:{c['on_primary']} !important}}"
            "[data-testid='stBaseButton-primary'] *,[data-testid='stBaseButton-primaryFormSubmit'] *"
            f"{{color:{c['on_primary']} !important}}"
            "[data-testid='stBaseButton-primary']:hover,[data-testid='stBaseButton-primaryFormSubmit']:hover"
            f"{{background:{c['primary_deep']} !important;border-color:{c['primary_deep']} !important}}"
            "</style>")
