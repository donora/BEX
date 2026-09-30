"""Plausibility profiles — swappable regional species lists (PLAN.md §3c, D4).

A profile answers "which species are plausible at this site, and how plausible":

    profiles/<name>/
    ├── profile.toml    # display name, region, source, citation, date
    └── tiers.csv       # species_key, tier[, anything else — extras are kept]

Tiers: 0 = on the national/regional list, 1 = regular (breeding/wintering/passage),
2 = plausible vagrant, 3 = implausible. Species *absent* from a profile are treated
as tier 3 by convention (`Profile.tier_of`), so a profile only needs to enumerate
what's plausible, not the world.

Designed for other people's lists from the start: loading validates loudly
(unknown species *reported*, never dropped), and `save_profile` round-trips so the
app's upload flow can write what it read.
"""
from __future__ import annotations

import csv
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .taxonomy import normalise_key

IMPLAUSIBLE_TIER = 3
VALID_TIERS = (0, 1, 2, 3)


class ProfileError(ValueError):
    """A profile directory violated the format. Message says exactly how."""


@dataclass
class Profile:
    name: str                       # directory name; the identifier in manifests
    display_name: str
    region: str = ""
    source: str = ""
    citation: str = ""
    date: str = ""
    tiers: dict[str, int] = field(default_factory=dict)   # species_key -> tier
    extras: dict[str, dict[str, str]] = field(default_factory=dict)  # key -> extra cols

    def tier_of(self, species_key: str) -> int:
        """Tier for any species; absent from the list means implausible (tier 3)."""
        return self.tiers.get(species_key, IMPLAUSIBLE_TIER)

    def plausible_keys(self, max_tier: int = 1) -> set[str]:
        return {k for k, t in self.tiers.items() if t <= max_tier}


def load_profile(profile_dir: str | Path) -> Profile:
    d = Path(profile_dir)
    meta_path, tiers_path = d / "profile.toml", d / "tiers.csv"
    problems = [f"missing {p.name}" for p in (meta_path, tiers_path) if not p.exists()]
    if problems:
        raise ProfileError(f"{d}: {', '.join(problems)}")

    with open(meta_path, "rb") as f:
        meta = tomllib.load(f).get("profile", {})
    if not meta.get("display_name"):
        raise ProfileError(f"{meta_path}: [profile].display_name is required")

    tiers: dict[str, int] = {}
    extras: dict[str, dict[str, str]] = {}
    with open(tiers_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "species_key" not in cols or "tier" not in cols:
            raise ProfileError(f"{tiers_path}: needs columns species_key,tier (got {cols})")
        for i, row in enumerate(reader, start=2):
            key = normalise_key(row["species_key"])
            if not key:
                raise ProfileError(f"{tiers_path}:{i}: empty species_key")
            try:
                tier = int(row["tier"])
            except (TypeError, ValueError):
                raise ProfileError(f"{tiers_path}:{i}: tier {row['tier']!r} is not an integer")
            if tier not in VALID_TIERS:
                raise ProfileError(f"{tiers_path}:{i}: tier {tier} not in {VALID_TIERS}")
            if key in tiers:
                raise ProfileError(f"{tiers_path}:{i}: duplicate species_key {key!r}")
            tiers[key] = tier
            extra = {c: row[c] for c in cols if c not in ("species_key", "tier") and row[c]}
            if extra:
                extras[key] = extra

    return Profile(
        name=d.name,
        display_name=meta["display_name"],
        region=meta.get("region", ""),
        source=meta.get("source", ""),
        citation=meta.get("citation", ""),
        date=meta.get("date", ""),
        tiers=tiers,
        extras=extras,
    )


def save_profile(profiles_dir: str | Path, profile: Profile) -> Path:
    d = Path(profiles_dir) / profile.name
    d.mkdir(parents=True, exist_ok=True)
    toml = (
        "[profile]\n"
        f'display_name = "{profile.display_name}"\n'
        f'region = "{profile.region}"\n'
        f'source = "{profile.source}"\n'
        f'citation = "{profile.citation}"\n'
        f'date = "{profile.date}"\n'
    )
    (d / "profile.toml").write_text(toml)

    extra_cols = sorted({c for e in profile.extras.values() for c in e})
    with open(d / "tiers.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["species_key", "tier", *extra_cols])
        for key in sorted(profile.tiers):
            extra = profile.extras.get(key, {})
            writer.writerow([key, profile.tiers[key], *[extra.get(c, "") for c in extra_cols]])
    return d


def list_profiles(profiles_dir: str | Path) -> list[str]:
    d = Path(profiles_dir)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "profile.toml").exists())


def unknown_species(profile: Profile, canonical_keys: set[str]) -> list[str]:
    """Species in the profile that no loaded vocabulary knows — reported, not dropped.

    A non-empty result is not necessarily an error: the canonical table only grows
    as model label files are ingested (Stages 2/5), and a UK profile is *expected*
    to be full of species the SNE-only table has never heard of. The app shows this
    list on upload; stats code treats unknown-but-listed species as plausible.
    """
    return sorted(k for k in profile.tiers if k not in canonical_keys)
