# Tour screenshots

`make_tour.py` regenerates the annotated screenshots in `docs/tour/` that
TOUR.md uses: it starts the app, drives it in Chrome, crops the captures and
draws the numbered markers.

```bash
cd scripts/tour && npm install playwright-core && cd ../..
.venv/bin/python scripts/tour/make_tour.py
```

Needs the SNE demo dataset (with its model runs and alignments), Node.js and
Google Chrome. Temporary demo labels and a demo setup are made and removed;
nothing else in the store is changed.

Markers and crops are placed by element (`SHOTS` in `make_tour.py`), measured
in the browser as each page is captured, so they follow layout changes. A
marker whose element is missing is reported and left out; the raw captures
stay in `raw/`. Screenshots not listed in `SHOTS` (Scorecard, Geofilter, older
Survey and Thresholds images) are not regenerated.
