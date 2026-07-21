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

- **The Caddy proxy with automatic HTTPS (recommended for public use)** — see §5.
- **An SSH tunnel** for a quick private look:
  `ssh -L 8000:127.0.0.1:8000 <user>@<SERVER_IP>` then open `http://127.0.0.1:8000/`.
- **Direct public HTTP (no TLS):**
  `APP_BIND=0.0.0.0 docker compose up -d` → `http://<SERVER_IP>:8000/`.

The app is intentionally public (no login), so serving it openly is expected —
§5 just adds a domain + HTTPS, which browsers want.

Logs:

```bash
docker compose logs -f app
```

---

## Step 5 — Public HTTPS on a domain (Caddy, automatic TLS)

The app is intentionally public (no login). HTTPS is done with **Caddy** (auto
Let's Encrypt cert — no certbot, no cert files, no renewal cron).

**Only one process can bind ports 80/443 on a host.** So choose the mode that
matches the server:

- **Mode A — standalone** (this app is the only thing on the VPS): use the
  bundled Caddy in `docker-compose.yml` (the `proxy` profile). See §5A.
- **Mode B — shared reverse proxy** (this app runs *alongside another app*, e.g.
  OTN, that already owns 80/443): do **not** run the bundled Caddy. Attach the
  app to a shared network and add one site block to the shared Caddyfile. See §5B.
  **This is the mode for `otn.sebastiankrantz.com`'s server.**

### 5a — Point DNS at the server (both modes)

In your DNS provider (Namecheap), create an **A record**. Multiple A records may
point at the same IP — `infrarisk` alongside `otn` on one server is normal
name-based virtual hosting, not a conflict:

```
infrarisk.sebastiankrantz.com.   A   <SERVER_IP>
```

(Add an `AAAA` record too if the VPS has an IPv6 address.) Caddy cannot get a
certificate until this resolves, so verify first:

```bash
dig +short infrarisk.sebastiankrantz.com    # must return <SERVER_IP>
```

### 5b — Open the firewall (both modes)

The front-door Caddy needs public **80 and 443** (80 is required for the ACME
HTTP challenge and the HTTP→HTTPS redirect). If OTN's Caddy is already serving,
these are already open. On Ubuntu (UFW):

```bash
sudo apt-get install -y ufw
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
sudo ufw status
```

> **Hostinger:** there is also a **panel-level firewall** (hPanel → VPS →
> Firewall) that is independent of UFW. Make sure it allows 22/80/443.

### 5A — Mode A: standalone (bundled Caddy owns 80/443)

Use only if nothing else on the host is on 80/443.

```bash
docker compose --profile proxy up -d --build
```

The domain defaults to `infrarisk.sebastiankrantz.com` (baked into
`docker-compose.yml`); override with `DOMAIN=... docker compose ...`. Caddy
issues the cert on first start and serves `https://infrarisk.sebastiankrantz.com/`,
redirecting HTTP→HTTPS. Certs persist in the `caddy_data` volume.

### 5B — Mode B: behind a shared reverse proxy (multi-app server)

Use when a front-door Caddy already owns 80/443. The app runs as a **backend
only** — its container listens on **port 8000** and joins a shared Docker network
so the front-door Caddy can route to it.

**On `sebastiankrantz.com`'s server** the front door is the dedicated **edge
Caddy stack at `/opt/edge`** (container `edge-caddy`, on network `web`), which
also serves `otn.sebastiankrantz.com`. The commands below assume that stack. If
your proxy lives elsewhere or is named differently, substitute its path,
container name, and Caddyfile location.

1. Ensure the shared network exists (the edge stack already created it):

   ```bash
   docker network create web    # once; skip if it already exists (it does here)
   ```

2. Start the app with the shared overlay — **no `proxy` profile**:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.shared.yml up -d --build
   ```

3. Add a site block to the **edge** Caddyfile — `/opt/edge/Caddyfile` (note:
   **port 8000**):

   ```
   infrarisk.sebastiankrantz.com {
       reverse_proxy infrarisk-app:8000
   }
   ```

4. Zero-downtime reload the edge Caddy (no restart, other apps untouched):

   ```bash
   docker compose exec edge-caddy caddy reload --config /etc/caddy/Caddyfile
   # run from /opt/edge, or: docker exec edge-caddy caddy reload --config /etc/caddy/Caddyfile
   ```

`edge-caddy` and `infrarisk-app` are both on `web` (the overlay in step 2 joins
`infrarisk-app`), so no `docker network connect` is needed. Confirm with
`docker network inspect web` if a route ever 502s.

### 5c — Verify (both modes)

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
| Caddy TLS fails / no cert | DNS A record not resolving yet, or ports 80/443 blocked (§5b) — `docker compose logs caddy` |
| Caddy: "too many certificates" | Hit Let's Encrypt rate limit from repeated re-issue; wait, or test with `DOMAIN` on a staging cert |

---

## File map

| Path | Role |
|------|------|
| `deploy/DEPLOY.md` | **This runbook — start here** |
| `Dockerfile` | Multi-stage Node build + Python 3.11 app |
| `docker-compose.yml` | `app` (+ optional bundled `caddy` proxy profile) |
| `docker-compose.shared.yml` | Overlay for Mode B: app joins external `web` net, no own Caddy |
| `scripts/download_hazard_rasters.py` | Download COGs + rewrite CSV for Docker |
| `data/download_rasters.R` | Same URL list as CSV (R downloader; no CSV rewrite) |
| `data/hazard_layers.csv` | Hazard catalog (URLs rewritten on server after download) |
| `deploy/Caddyfile` | Bundled reverse proxy + automatic HTTPS (Mode A only) |

---

## Checklist (agent)

- [ ] Cloned `docker-deployment` from https://github.com/SebKrantz/InfraRiskApp.git
- [ ] Docker Engine + Compose available
- [ ] Ran `python3 scripts/download_hazard_rasters.py --path-prefix /app/data/rasters`
- [ ] Verification script reports all rasters present
- [ ] `docker compose up -d --build` and `/health` is healthy (bound to 127.0.0.1)
- [ ] UI smoke test passed (via SSH tunnel or the proxy)
- [ ] If public: A record for `infrarisk.sebastiankrantz.com` → server IP resolves
- [ ] If public: UFW (and Hostinger panel firewall) allow 22/80/443
- [ ] Picked proxy mode: **A** (standalone, own Caddy) or **B** (shared Caddy already on 80/443)
- [ ] Mode A: `docker compose --profile proxy up -d --build`
- [ ] Mode B: `docker compose -f docker-compose.yml -f docker-compose.shared.yml up -d --build`; add `infrarisk-app:8000` block to `/opt/edge/Caddyfile`; `docker compose exec edge-caddy caddy reload --config /etc/caddy/Caddyfile`
- [ ] HTTPS `/health` healthy at `https://infrarisk.sebastiankrantz.com`
