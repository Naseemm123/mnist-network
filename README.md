# MNIST from Scratch

An end-to-end MNIST classifier built with NumPy and a static React visualizer. The browser performs the same `784 → 128 → 10` ReLU/softmax forward pass locally, with no inference server or web ML framework.

## Repository layout

- `model/` — NumPy training code and JSON weight exporter.
- `web/` — Vite + React static application for GitHub Pages.
- `.github/workflows/deploy.yml` — build and Pages deployment workflow.

## Train and export

From `model/`, install the Python dependencies and run:

```bash
python main.py
```

After training, `main.py` writes `model/weights.json` and mirrors it to `web/public/weights.json`.

## Run the frontend

```bash
cd web
npm install
npm run dev
```

The checked-in weights are deterministic development weights so the UI can be previewed before a training run. Replace them by running the exporter for the trained model.
