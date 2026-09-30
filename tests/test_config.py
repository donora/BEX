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
