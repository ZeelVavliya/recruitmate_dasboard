# Placement Survival Lab

Mock-data pipeline for the thriving and retention approach. It simulates agency placements with a known truth, fits a discrete-time competing-risks survival model, checks the model against that truth and replays the same pipeline on public turnover data. Background: [../DATA_AND_CHECKINS.md](../DATA_AND_CHECKINS.md).

| File | What it is |
|---|---|
| `app.py` | Streamlit dashboard. Re-simulates live from the sidebar assumptions; two extra tabs show the software-engineer results (thriving model comparison, a thriving calculator, retention model comparison) |
| `sim.py` / `pipeline.py` / `replay.py` | Simulator, model and evaluation, public-data loaders |
| `sim.ipynb` | Walkthrough notebook; its last cell refreshes `dashboard.html` |
| `dashboard.html` | Static version of the dashboard (no Python needed, no live re-simulation) |
| `swe_models.py` | Software-engineer thriving models on the Stack Overflow 2019 survey (downloads it, ~200 MB) |
| `swe_retention.py` | Software-engineer retention: six survival models compared on mock placements |

## Run locally

```
cd mock
pip install -r requirements.txt
streamlit run app.py
```

The first load takes about 10 seconds while it simulates and fits. After that, each set of assumptions is cached.

## Deploy on Vercel

Vercel runs the app as a container: it detects `Dockerfile.vercel` and routes all traffic to it.

**With the CLI** (no Git needed):

```
npm i -g vercel
cd mock
vercel          # first run creates the project
vercel --prod
```

**From GitHub:** push the repo, choose **Import Project** in Vercel and set **Root Directory** to `mock`.

Know before you rely on it:
- Container images and WebSockets on Vercel Functions are both in **beta** (as of September 2026). Streamlit needs both.
- A WebSocket closes when the function reaches its **maximum duration**. Streamlit then shows "Connecting…" for a moment and reruns the page, which may reset the sidebar.
- Instances scale down after 5 idle minutes. The next visit waits for the container to start and the models to fit (about 10–20 seconds).
- Vercel sends traffic to port 80 unless you set a `PORT` environment variable in the project. The container listens on `$PORT` or 80.
- The container build hasn't been tested locally (Docker wasn't running). The app itself was tested with Streamlit's AppTest and in a browser.

**If Vercel misbehaves**, use Streamlit Community Cloud (free, built for this): share.streamlit.io → **Create app** → your repo, main file `mock/app.py`, Python 3.12 under Advanced settings. It installs `requirements.txt` from next to the app.

**If you only need the results on Vercel**, deploy `dashboard.html` as a static site: copy it to an empty folder as `index.html` and run `vercel` there. It needs no Python and never cold-starts, but it can't re-simulate.

## Licences

The public datasets are non-commercial: Babushkin Employee Turnover (CC BY-NC-SA 4.0) and Huebner & Patalano HR Data Set v13 (CC BY-NC-ND 4.0). A demo deployment is fine; remove the Replay tab before this becomes part of a paid product.
