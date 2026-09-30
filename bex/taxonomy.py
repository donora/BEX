"""Species identity — the unglamorous module that makes everything joinable.

Three label vocabularies collide in this project (PLAN.md §3c): SNE annotations use
eBird codes, BirdNET uses `Genus species_Common Name` strings, Perch uses eBird
codes from a different taxonomy year. A naive string join silently drops species
and fabricates disagreements between models — so every vocabulary maps into one
canonical key, and every mapping produces an explicit **unmapped report** instead
of quiet loss.

The canonical `species_key` is the scientific name, normalised: `"Turdus merula"`.
Scientific names are the most stable currency across model vintages and national
lists; where taxonomies genuinely diverge (splits/lumps), per-model synonym tables
are the fix, added when a real unmapped report demands them — not preemptively.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


def normalise_key(name: str) -> str:
    """'  turdus   MERULA ' -> 'Turdus merula'. The one true spelling of a key."""
    parts = name.strip().split()
    if not parts:
        return ""
    genus = parts[0].capitalize()
    rest = [p.lower() for p in parts[1:]]
    return " ".join([genus, *rest])


_SPECIES_RE = re.compile(r"^[A-Z][a-z]+ [a-z\-]+( [a-z\-]+)?$")


def is_species(key: str) -> bool:
    """A named species ("Turdus migratorius"), not a sound-event class. Perch also
    scores ~80 AudioSet classes ("Car", "Church_bell") and BirdNET a few ("Dog",
    "Engine"); none is a Latin binomial, and none belongs on a species list."""
    return bool(_SPECIES_RE.match(key))


def genus(species_key: str) -> str:
    return species_key.split()[0] if species_key else ""


def xc_species_url(species_key: str) -> str:
    """Xeno-Canto species page (PLAN.md §7 layer 1) — stable, keyless, world-wide.

    e.g. 'Turdus merula' -> https://xeno-canto.org/species/Turdus-merula
    """
    return "https://xeno-canto.org/species/" + species_key.replace(" ", "-")


# --------------------------------------------------------------------------- #
# Canonical table
# --------------------------------------------------------------------------- #

CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "species_key": pl.Utf8,
    "common_name": pl.Utf8,
    "ebird_code": pl.Utf8,  # "" when we don't know one
}


def canonical_from_sne(labels_dir: str | Path) -> pl.DataFrame:
    """Build canonical rows from the SNE dataset's own species.csv.

    This is the labelled test rig's vocabulary (56 species). Model adapters extend
    the canonical table with their own label files in Stages 2/5; the SNE table is
    what Stage 0's losslessness deliverable is checked against.
    """
    raw = pl.read_csv(Path(labels_dir) / "species.csv")
    df = raw.rename(
        {
            "Species eBird Code": "ebird_code",
            "Scientific Name": "species_key",
            "Common Name": "common_name",
        }
    ).with_columns(
        pl.col("species_key").map_elements(normalise_key, return_dtype=pl.Utf8)
    )
    return df.select(list(CANONICAL_SCHEMA)).sort("species_key")


# --------------------------------------------------------------------------- #
# Label mapping, with the unmapped report as a first-class result
# --------------------------------------------------------------------------- #

@dataclass
class MappingReport:
    """The result of mapping one model's label list into canonical keys.

    `mapping` covers every label that resolved; `unmapped` is every label that
    did not, verbatim, so it can be shown to a human. An empty `unmapped` is the
    losslessness deliverable; a non-empty one is a to-do list, never a silent drop.
    """

    source: str
    mapping: dict[str, str] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)

    @property
    def lossless(self) -> bool:
        return not self.unmapped

    def summary(self) -> str:
        head = f"{self.source}: {len(self.mapping)} mapped, {len(self.unmapped)} unmapped"
        if self.unmapped:
            shown = ", ".join(self.unmapped[:10])
            more = "" if len(self.unmapped) <= 10 else f" (+{len(self.unmapped) - 10} more)"
            return f"{head} — [{shown}{more}]"
        return head


def map_ebird_codes(codes: list[str], canonical: pl.DataFrame, source: str = "ebird") -> MappingReport:
    """eBird code -> species_key via the canonical table."""
    lookup = dict(
        canonical.filter(pl.col("ebird_code") != "")
        .select("ebird_code", "species_key")
        .iter_rows()
    )
    report = MappingReport(source=source)
    for code in codes:
        key = lookup.get(code)
        if key is None:
            report.unmapped.append(code)
        else:
            report.mapping[code] = key
    return report


def map_scientific_names(
    labels: list[str],
    canonical: pl.DataFrame | None = None,
    source: str = "scientific",
    split_common: bool = True,
) -> MappingReport:
    """BirdNET-style labels ('Genus species_Common Name') or bare scientific names.

    With a canonical table, unknown names are reported unmapped; without one
    (canonical=None) every parseable name normalises to a key — used when a model's
    label file *defines* new canonical rows rather than joining existing ones.

    `split_common` says whether an underscore separates a trailing common name.
    It must be set per model rather than guessed: BirdNET writes
    'Turdus merula_Eurasian Blackbird', while Perch writes bare scientific names
    *and* AudioSet-style event classes like 'Bass_drum' and 'Bass_guitar' —
    splitting those would collapse both to 'Bass' and silently misalign every
    score column after them.
    """
    known = set(canonical["species_key"]) if canonical is not None else None
    report = MappingReport(source=source)
    for label in labels:
        sci = label.split("_", 1)[0] if split_common else label
        key = normalise_key(sci)
        if not key or (known is not None and key not in known):
            report.unmapped.append(label)
        else:
            report.mapping[label] = key
    return report
