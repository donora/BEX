"""Starred datasets: the small state the Home page keeps."""
from bex import prefs


def test_stars_toggle_and_persist(tmp_path):
    assert prefs.starred(tmp_path) == []
    prefs.toggle_star(tmp_path, "a")
    prefs.toggle_star(tmp_path, "b")
    assert prefs.starred(tmp_path) == ["a", "b"]
    prefs.toggle_star(tmp_path, "a")
    assert prefs.starred(tmp_path) == ["b"]


def test_starred_datasets_come_first_and_each_group_keeps_its_order():
    assert prefs.stars_first(["a", "b", "c", "d"], ["d", "b"]) == ["b", "d", "a", "c"]
    assert prefs.stars_first(["a", "b"], ["gone"]) == ["a", "b"]
