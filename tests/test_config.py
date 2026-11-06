from bex.config import load_config


def test_example_config_loads(repo):
    cfg = load_config(repo / "bex.example.toml")
    assert cfg.source == repo / "bex.example.toml"
    # Relative paths resolve against the config file's directory.
    assert cfg.store_dir == repo / "store"
    assert cfg.profiles_dir == repo / "profiles"
    assert cfg.audio_dir == repo.parent / "data" / "audio"
    assert cfg.site_name == "sierra-nevada"
    assert cfg.site_lat == 38.49
    assert cfg.default_profile == "uk-starter"


def test_explicit_config(tmp_path):
    p = tmp_path / "bex.toml"
    p.write_text('[paths]\nstore_dir = "/tmp/abs"\ncache_dir = "rel"\n')
    cfg = load_config(p)
    assert str(cfg.store_dir) == "/tmp/abs"      # absolute stays absolute
    assert cfg.cache_dir == tmp_path / "rel"     # relative resolves to the toml dir
    assert cfg.xc_api_key == ""


def test_display_section_is_read_and_optional(tmp_path):
    p = tmp_path / "bex.toml"
    p.write_text("[display]\nfmin = 60\n")
    assert load_config(p).display == {"fmin": 60}
    p.write_text("")
    assert load_config(p).display == {}


def test_default_threshold_set_is_read_and_optional(tmp_path):
    p = tmp_path / "bex.toml"
    p.write_text('[defaults]\nthreshold_set = "demo"\n')
    assert load_config(p).default_threshold_set == "demo"
    p.write_text("")
    assert load_config(p).default_threshold_set == ""


def test_project_defaults_terms_and_links_are_read_and_optional(tmp_path):
    p = tmp_path / "bex.toml"
    p.write_text('[defaults]\ndataset = "pond"\n[terms]\nbirdsong = "frog call"\n'
                 "[xeno_canto]\nlinks = false\n")
    c = load_config(p)
    assert (c.default_dataset, c.terms, c.xc_links) == ("pond", {"birdsong": "frog call"}, False)
    p.write_text("")
    c = load_config(p)
    assert (c.default_dataset, c.terms, c.xc_links) == ("", {}, True)


def test_species_first_is_read_and_optional(tmp_path):
    p = tmp_path / "bex.toml"
    p.write_text('[defaults]\nspecies_first = ["a", "b"]\n')
    assert load_config(p).species_first == ("a", "b")
    p.write_text("")
    assert load_config(p).species_first == ()
