import pytest

from bex.profiles import (
    IMPLAUSIBLE_TIER,
    Profile,
    ProfileError,
    list_profiles,
    load_profile,
    save_profile,
    unknown_species,
)
from bex.taxonomy import canonical_from_sne


def test_shipped_profiles_load(repo):
    names = list_profiles(repo / "profiles")
    assert names == ["uk-starter", "us-ca-sierra"]
    for name in names:
        p = load_profile(repo / "profiles" / name)
        assert p.display_name
        assert len(p.tiers) > 50
        assert set(p.tiers.values()) <= {0, 1, 2, 3}


def test_sierra_profile_covers_sne(repo, sne_labels_dir):
    """Every annotated SNE species is plausible in the Sierra profile — that's the
    property Stage 4's false-suppression check relies on."""
    canonical = canonical_from_sne(sne_labels_dir)
    p = load_profile(repo / "profiles" / "us-ca-sierra")
    assert unknown_species(p, set(canonical["species_key"])) == []
    assert p.plausible_keys() == set(canonical["species_key"])


def test_tier_semantics(repo):
    p = load_profile(repo / "profiles" / "uk-starter")
    assert p.tier_of("Turdus merula") == 1
    # Absent species are implausible by convention — the Australian blackbird case.
    assert p.tier_of("Strepera graculina") == IMPLAUSIBLE_TIER
    assert "Turdus merula" in p.plausible_keys()


def test_roundtrip(tmp_path):
    original = Profile(
        name="my-patch",
        display_name="My local patch",
        region="Perthshire",
        source="personal records",
        citation="",
        date="2026-09-07",
        tiers={"Turdus merula": 1, "Lophophanes cristatus": 0},
        extras={"Turdus merula": {"common_name": "Eurasian Blackbird"}},
    )
    save_profile(tmp_path, original)
    loaded = load_profile(tmp_path / "my-patch")
    assert loaded == original


def test_validation_errors(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    with pytest.raises(ProfileError, match="missing profile.toml"):
        load_profile(d)

    (d / "profile.toml").write_text('[profile]\ndisplay_name = "Broken"\n')
    (d / "tiers.csv").write_text("species_key,tier\nTurdus merula,9\n")
    with pytest.raises(ProfileError, match="tier 9"):
        load_profile(d)

    (d / "tiers.csv").write_text("species_key,tier\nTurdus merula,1\nturdus  merula,2\n")
    with pytest.raises(ProfileError, match="duplicate"):
        load_profile(d)  # normalisation makes these the same key — caught, not lost


def test_unknown_species_reported_not_dropped():
    p = Profile(name="x", display_name="X", tiers={"Imaginarius birdus": 1})
    report = unknown_species(p, {"Turdus merula"})
    assert report == ["Imaginarius birdus"]
    # ...and the species is still in the profile, still plausible.
    assert p.tier_of("Imaginarius birdus") == 1
