# Runner environments

Each model adapter runs as a CLI in its **own** virtual environment (PLAN.md §3b),
because BirdNET (TensorFlow), Perch (TF/JAX) and any torch-based arm cannot share
pins with each other or with the app. The app env never imports a model; adapters
communicate with the rest of the system only through the score store on disk.

Convention, per adapter:

```
envs/<model>/
├── requirements.txt # that adapter's pins — and nothing shared with the app
├── .venv/           # gitignored
└── run.py           # CLI: audio in -> score store + run manifest out
```

Each env installs the `bex` library itself with `--no-deps` — from this clone
(`pip install -e ../.. --no-deps`) or from PyPI (`pip install bex-bioacoustics
--no-deps`) — so none of the app's dependencies reach it. A runner finds your
project through `bex.toml`: run it from inside the project folder, or set
`BEX_CONFIG=/path/to/bex.toml`.
