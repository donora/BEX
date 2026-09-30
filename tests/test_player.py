"""The Explorer's audio player (bex.player)."""
import io

import numpy as np
import soundfile as sf

from bex.player import encode_mp3, player_html


def test_mp3_round_trips_at_the_same_rate_and_length():
    y = (np.sin(np.linspace(0, 2000, 32000 * 2)) * 0.3).astype(np.float32)
    z, sr = sf.read(io.BytesIO(encode_mp3(y, 32000)))
    assert sr == 32000
    assert abs(len(z) - len(y)) < 0.1 * 32000     # encoder padding only


def test_player_speaks_recording_time_and_lines_up_with_the_figure():
    html = player_html(b"x", 1920.0, 1980.0, 1920.0, (1950.0, 1955.0),
                       0.125, 0.9, [1920.0, 1940.0, 1960.0, 1980.0, 2000.0])
    assert ">1940</span>" in html and ">2000<" not in html   # ticks in view only
    assert "▶ window 1950–1955 s" in html
    assert "left:50.000%;width:8.333%" in html                  # the window band
    assert "padding:0 10.000% 0 12.500%" in html                # the figure's margins
    assert "data:audio/mpeg;base64," in html


def test_player_without_a_window_has_no_window_button():
    html = player_html(b"x", 0.0, 60.0, 0.0, None, 0.1, 0.9, [])
    assert 'id="pw"' not in html
