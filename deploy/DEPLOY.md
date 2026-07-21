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

On hosts where Docker ships pre-installed (e.g. Hostinger's Ubuntu 24.04 + Docker
template), this whole block is skipped and you only confirm the versions.

```bash
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  # Apply group without re-login when possible:
  exec sg docker newgrp docker
fi
docker --version
docker compose version   # must print v2.x — the Compose plugin is required
```

If `docker compose version` fails on a pre-installed host, the Compose plugin is
missing — install it with `sudo apt-get install -y docker-compose-plugin`.

If `docker compose` fails with permission errors, re-run under `sg docker -c '...'`
or have the operator re-login after `usermod -aG docker`. (Running as `root`, as is
common on a fresh VPS, avoids the docker-group step entirely.)

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

**The app binds to `127.0.0.1:8000` on the host by default — it is NOT reachable
from the internet.** This is deliberate: the app has no authentication and all
users share one in-memory upload state. To reach the UI, use one of:

- **The nginx proxy with TLS + basic auth (recommended)** — see §5.
- **An SSH tunnel** for a quick private look:
  `ssh -L 8000:127.0.0.1:8000 <user>@<SERVER_IP>` then open `http://127.0.0.1:8000/`.
- **Direct public exposure (NOT recommended, no auth):**
  `APP_BIND=0.0.0.0 docker compose up -d` → `http://<SERVER_IP>:8000/`.

Logs:

```bash
docker compose logs -f app
```

---

## Step 5 — Public HTTPS on a domain (do this for any public VPS)

The app has **no login and shared state**, so never leave port 8000 open to the
world. Put it behind the bundled nginx proxy (TLS + HTTP basic auth) on the
domain **`infrarisk.sebastiankrantz.com`**.

### 5a — Point DNS at the server

In your DNS provider, create an **A record**:

```
infrarisk.sebastiankrantz.com.   A   <SERVER_IP>
```

(Add a `AAAA` record too if the VPS has an IPv6 address.) Wait for it to
resolve before requesting a certificate:

```bash
dig +short infrarisk.sebastiankrantz.com    # must return <SERVER_IP>
```

### 5b — Open the firewall

Expose only SSH + HTTP/HTTPS; keep 8000 private. On Ubuntu (UFW):

```bash
sudo apt-get install -y ufw
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
sudo ufw status
```

> **Hostinger:** there is also a **panel-level firewall** (hPanel → VPS →
> Firewall) that is independent of UFW and cannot be set from this runbook. Make
> sure it allows 22/80/443 and does **not** allow 8000.

### 5c — Obtain a TLS certificate (Let's Encrypt)

The nginx container reads certs from `deploy/certs/`. Issue a cert on the host
with certbot in standalone mode (port 80 must be free — do this *before*
starting the proxy):

```bash
sudo apt-get install -y certbot
sudo certbot certonly --standalone -d infrarisk.sebastiankrantz.com \
  --agree-tos -m basti.krantz@gmail.com --no-eff-email

# Copy the issued cert into the path nginx.conf expects:
mkdir -p deploy/certs
sudo cp /etc/letsencrypt/live/infrarisk.sebastiankrantz.com/fullchain.pem deploy/certs/fullchain.pem
sudo cp /etc/letsencrypt/live/infrarisk.sebastiankrantz.com/privkey.pem   deploy/certs/privkey.pem
sudo chown "$USER":"$USER" deploy/certs/*.pem
```

Let's Encrypt certs expire after 90 days. To renew, re-run `certbot renew`
(stop the nginx container first so port 80 is free), re-copy the two `.pem`
files, and `docker compose --profile proxy restart nginx`.

### 5d — Create the basic-auth user

```bash
sudo apt-get install -y apache2-utils
htpasswd -c deploy/htpasswd analyst    # choose a strong password; do not commit
```

### 5e — Start the proxy

```bash
docker compose --profile proxy up -d --build
```

nginx now serves `https://infrarisk.sebastiankrantz.com/` on ports 80/443 per
`deploy/nginx.conf` (HTTP redirects to HTTPS; `/health` is exempt from auth).
Verify:

```bash
curl -fsS https://infrarisk.sebastiankrantz.com/health   # {"status":"healthy"}
```

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
- [ ] `docker compose up -d --build` and `/health` is healthy (bound to 127.0.0.1)
- [ ] UI smoke test passed (via SSH tunnel or the proxy)
- [ ] If public: A record for `infrarisk.sebastiankrantz.com` → server IP resolves
- [ ] If public: UFW (and Hostinger panel firewall) allow only 22/80/443
- [ ] If public: TLS cert in `deploy/certs/` + `deploy/htpasswd` created
- [ ] If public: `docker compose --profile proxy up -d --build`, HTTPS `/health` healthy
