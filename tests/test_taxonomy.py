import polars as pl

from bex.taxonomy import (
    canonical_from_sne,
    genus,
    map_ebird_codes,
    map_scientific_names,
    normalise_key,
    xc_species_url,
)


def test_normalise_key():
    assert normalise_key("  turdus   MERULA ") == "Turdus merula"
    assert normalise_key("Turdus merula") == "Turdus merula"
    assert normalise_key("") == ""


def test_genus():
    assert genus("Turdus merula") == "Turdus"
    assert genus("") == ""


def test_xc_url():
    assert xc_species_url("Turdus merula") == "https://xeno-canto.org/species/Turdus-merula"


def test_sne_canonical_lossless(sne_labels_dir):
    """Stage 0 deliverable: the taxonomy join is lossless on SNE's 56 species."""
    canonical = canonical_from_sne(sne_labels_dir)
    assert len(canonical) == 56
    assert canonical["species_key"].n_unique() == 56
    assert (canonical["ebird_code"] != "").all()

    # Every eBird code used in the annotations resolves to a canonical key.
    ann = pl.read_csv(sne_labels_dir / "annotations.csv")
    codes = sorted(ann["Species eBird Code"].unique())
    report = map_ebird_codes(codes, canonical)
    assert report.lossless, report.summary()
    assert set(report.mapping.values()) <= set(canonical["species_key"])


def test_unmapped_is_reported_not_dropped(sne_labels_dir):
    canonical = canonical_from_sne(sne_labels_dir)
    report = map_ebird_codes(["amerob", "notacode"], canonical)
    assert report.mapping["amerob"] == "Turdus migratorius"
    assert report.unmapped == ["notacode"]
    assert not report.lossless
    assert "notacode" in report.summary()


def test_birdnet_style_labels():
    canonical = pl.DataFrame(
        {
            "species_key": ["Turdus merula", "Strepera graculina"],
            "common_name": ["Eurasian Blackbird", "Pied Currawong"],
            "ebird_code": ["eurbla", "piecur1"],
        }
    )
    labels = ["Turdus merula_Eurasian Blackbird", "Gymnorhina tibicen_Australian Magpie"]
    report = map_scientific_names(labels, canonical, source="birdnet")
    assert report.mapping == {"Turdus merula_Eurasian Blackbird": "Turdus merula"}
    assert report.unmapped == ["Gymnorhina tibicen_Australian Magpie"]

    # Without a canonical table, labels *define* keys (used to ingest label files).
    open_report = map_scientific_names(labels, canonical=None)
    assert open_report.lossless
    assert open_report.mapping[labels[1]] == "Gymnorhina tibicen"
