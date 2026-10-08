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


def save_profile(profiles_dir: str | Path, profile: Profile,
                 bbox: tuple[float, float, float, float] | None = None) -> Path:
    import json
    d = Path(profiles_dir) / profile.name
    d.mkdir(parents=True, exist_ok=True)
    q = json.dumps   # a JSON string is a valid TOML basic string, quotes escaped
    toml = (
        "[profile]\n"
        f"display_name = {q(profile.display_name)}\n"
        f"region = {q(profile.region)}\n"
        f"source = {q(profile.source)}\n"
        f"citation = {q(profile.citation)}\n"
        f"date = {q(profile.date)}\n"
        + (f"bbox = [{', '.join(f'{v:g}' for v in bbox)}]\n" if bbox else "")
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


def profile_from_csv(text: str, name: str, display_name: str,
                     lookup: dict[str, str], region: str = "", source: str = "",
                     ) -> tuple[Profile, list[str]]:
    """An uploaded species list -> a Profile, and the names that matched no
    species (the caller refuses the upload while any remain: never dropped).

    Forgiving about the shape, strict about the content. The species column may
    be `species_key`, `scientific_name`, `species` or `common_name`; names are
    matched through `lookup` (lower-cased scientific and common names -> key),
    and a well-formed scientific name nobody knows is kept as written.
    `tier` is optional (1, regular, when missing); other columns are kept.
    """
    import io
    from datetime import date
    rows = list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))
    if not rows:
        raise ProfileError("the file has no rows")
    cols = list(rows[0])
    sp_col = next((c for c in ("species_key", "scientific_name", "species",
                               "common_name") if c in cols), None)
    if sp_col is None:
        raise ProfileError(f"no species column — expected species_key (or "
                           f"scientific_name / common_name); got {cols}")
    tiers, extras, unmapped = {}, {}, []
    for i, r in enumerate(rows, start=2):
        raw = (r.get(sp_col) or "").strip()
        if not raw:
            continue
        key = lookup.get(raw.lower())
        if key is None and sp_col != "common_name" and len(raw.split()) >= 2 \
                and raw[0].isupper():
            key = normalise_key(raw)          # a scientific name: keep it as written
        if key is None:
            unmapped.append(raw)
            continue
        t = (r.get("tier") or "").strip()
        try:
            tier = int(t) if t else 1
        except ValueError:
            raise ProfileError(f"row {i}: tier {t!r} is not 0, 1, 2 or 3")
        if tier not in VALID_TIERS:
            raise ProfileError(f"row {i}: tier {tier} is not 0, 1, 2 or 3")
        tiers[key] = min(tier, tiers.get(key, tier))
        extras[key] = {c: v for c, v in r.items()
                       if c not in (sp_col, "tier", "species_key") and v}
    if not tiers and not unmapped:
        raise ProfileError("no species found in the file")
    return Profile(name=name, display_name=display_name or name, region=region,
                   source=source or "uploaded in the app", date=date.today().isoformat(),
                   tiers=tiers, extras=extras), sorted(set(unmapped))


def profile_bbox(profile_dir: str | Path) -> tuple[float, float, float, float] | None:
    """The area a profile is meant for, if its profile.toml says:
    `bbox = [min_lat, min_lon, max_lat, max_lon]`. Optional — only used to
    suggest a profile for a new dataset's location."""
    p = Path(profile_dir) / "profile.toml"
    with open(p, "rb") as f:
        box = tomllib.load(f).get("profile", {}).get("bbox")
    if not box or len(box) != 4:
        return None
    return tuple(float(v) for v in box)


def suggest_profile(profiles_dir: str | Path, lat: float, lon: float) -> str | None:
    """The profile whose area contains (lat, lon) — the smallest, when several
    do — or None. A suggestion, never a silent default: the caller shows it."""
    hits = []
    for name in list_profiles(profiles_dir):
        box = profile_bbox(Path(profiles_dir) / name)
        if box and box[0] <= lat <= box[2] and box[1] <= lon <= box[3]:
            hits.append(((box[2] - box[0]) * (box[3] - box[1]), name))
    return min(hits)[1] if hits else None


def unknown_species(profile: Profile, canonical_keys: set[str]) -> list[str]:
    """Species in the profile that no loaded vocabulary knows — reported, not dropped.

    A non-empty result is not necessarily an error: the canonical table only grows
    as model label files are ingested (Stages 2/5), and a UK profile is *expected*
    to be full of species the SNE-only table has never heard of. The app shows this
    list on upload; stats code treats unknown-but-listed species as plausible.
    """
    return sorted(k for k in profile.tiers if k not in canonical_keys)
