# Agent instructions: deploy InfraRiskApp with Docker

**Audience:** An autonomous coding agent (e.g. Claude Code) on a remote Linux
server (Ubuntu 24.04 recommended). A human will give you only the repo URL and
tell you to deploy from branch `docker-deployment`. Follow this file end-to-end.

**Repo:** https://github.com/SebKrantz/InfraRiskApp.git  
**Branch:** `docker-deployment`  
**Do not** expect `data/rasters/` in git — GeoTIFFs are never committed. You must
download them onto the server before starting the containers.

---

## What you are deploying

Full-stack Infrastructure Risk Analyzer:

- FastAPI backend + built React SPA in one Docker image
- Hazard COGs bind-mounted from host `./data/rasters` (local disk, for speed)
- **Must run with a single uvicorn worker** (in-memory uploads/caches)
- No built-in authentication

---

## Step 0 — Preconditions

```bash
df -h .          # Need tens of GB free (full catalog is large; landslides alone are multi-GB)
free -h          # 16 GB RAM is comfortable
```

If free disk is under ~50 GB, warn the operator before downloading.

---

## Step 1 — Clone the branch

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git python3

git clone -b docker-deployment --single-branch \
  https://github.com/SebKrantz/InfraRiskApp.git InfraRiskApp
cd InfraRiskApp
```

If the repo already exists on the server:

```bash
cd InfraRiskApp
git fetch origin docker-deployment
git checkout docker-deployment
git pull --ff-only origin docker-deployment
```

---

## Step 2 — Install Docker

```bash
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  # Apply group without re-login when possible:
  exec sg docker newgrp docker
fi
docker --version
docker compose version
```

If `docker compose` fails with permission errors, re-run under `sg docker -c '...'`
or have the operator re-login after `usermod -aG docker`.

---

## Step 3 — Download hazard rasters to `data/rasters/` (required)

Rasters are **not** in the git repo. Download every COG from
`data/hazard_layers.csv` onto the server, then rewrite the CSV so the app uses
local paths inside the container (`/app/data/rasters/...`).

**Preferred (Python, no extra packages):**

```bash
cd /path/to/InfraRiskApp   # repo root

# Preview what will be fetched
python3 scripts/download_hazard_rasters.py --dry-run --path-prefix /app/data/rasters

# Download into data/rasters/ AND rewrite data/hazard_layers.csv for Docker
python3 scripts/download_hazard_rasters.py --path-prefix /app/data/rasters
```

This script:

1. Backs up remote URLs once → `data/hazard_layers.remote.csv`
2. Downloads each HTTPS GeoTIFF into `data/rasters/` (skips files already present)
3. Sets each `dataset_url` to `/app/data/rasters/<filename>.tif`

**Alternative (R only downloads; does not rewrite CSV):**

`data/download_rasters.R` lists the same hazard → URL pairs as
`data/hazard_layers.csv`. Use it only if you prefer R, then still run the Python
script afterward to rewrite paths:

```bash
Rscript data/download_rasters.R
python3 scripts/download_hazard_rasters.py --path-prefix /app/data/rasters
```

### Verify every layer has a local file

```bash
python3 - <<'PY'
import csv
from pathlib import Path
rasters = Path("data/rasters")
missing = []
with open("data/hazard_layers.csv", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        name = Path(row["dataset_url"]).name
        path = rasters / name
        ok = path.is_file() and path.stat().st_size > 0
        print(("OK  " if ok else "MISS"), name)
        if not ok:
            missing.append(name)
if missing:
    raise SystemExit(f"{len(missing)} missing raster(s): {missing}")
print("All rasters present.")
PY
```

**Do not start Docker until this check passes.**

---

## Step 4 — Build and run

```bash
docker compose up -d --build
curl -fsS http://127.0.0.1:8000/health
# expect: {"status":"healthy"}
```

App URL: `http://<SERVER_IP>:8000/`

Logs:

```bash
docker compose logs -f app
```

---

## Step 5 — Optional public HTTPS + basic auth

The app has no login. For a public VPS, enable the nginx profile:

```bash
sudo apt-get install -y apache2-utils
htpasswd -c deploy/htpasswd analyst    # choose a strong password; do not commit

# Place TLS certs as:
#   deploy/certs/fullchain.pem
#   deploy/certs/privkey.pem
# (certbot, Cloudflare origin cert, etc.)

docker compose --profile proxy up -d --build
```

Then serve on ports 80/443 per `deploy/nginx.conf`.

---

## Smoke test

1. Open the UI; hazard dropdown is populated.
2. Select a hazard — map tiles load (not endless spinner / HTTP 500).
3. Upload a small point/line GeoJSON or GPKG; run analysis; chart updates.
4. `docker compose restart app` clears in-memory uploads (expected).

---

## Architecture notes (do not “fix” these)

| Fact | Implication |
|------|-------------|
| `--workers 1` only | Never scale to multiple uvicorn workers/replicas without Redis |
| `./data` mounted read-only at `/app/data` | CSV + rasters live on the host; image rebuild does not refresh rasters |
| `data/rasters/` gitignored | Always download on the server (§3) |
| In-memory GeoDataFrames | Process restart loses uploads |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Empty hazard list | Confirm `data/hazard_layers.csv` exists and compose mounts `./data` |
| Tile/analysis GDAL errors | CSV still has `https://...` or file missing — re-run §3 |
| Permission denied reading rasters | `chmod -R a+rX data` |
| Docker permission denied | `sg docker -c 'docker compose up -d --build'` |
| OOM | Lower `GDAL_CACHEMAX` in compose env; ensure only one app container |

---

## File map

| Path | Role |
|------|------|
| `deploy/DEPLOY.md` | **This runbook — start here** |
| `Dockerfile` | Multi-stage Node build + Python 3.11 app |
| `docker-compose.yml` | `app` (+ optional `nginx` profile) |
| `scripts/download_hazard_rasters.py` | Download COGs + rewrite CSV for Docker |
| `data/download_rasters.R` | Same URL list as CSV (R downloader; no CSV rewrite) |
| `data/hazard_layers.csv` | Hazard catalog (URLs rewritten on server after download) |
| `deploy/nginx.conf` | TLS reverse proxy + basic auth |

---

## Checklist (agent)

- [ ] Cloned `docker-deployment` from https://github.com/SebKrantz/InfraRiskApp.git
- [ ] Docker Engine + Compose available
- [ ] Ran `python3 scripts/download_hazard_rasters.py --path-prefix /app/data/rasters`
- [ ] Verification script reports all rasters present
- [ ] `docker compose up -d --build` and `/health` is healthy
- [ ] UI smoke test passed
- [ ] (Optional) TLS + `deploy/htpasswd` if exposing publicly
