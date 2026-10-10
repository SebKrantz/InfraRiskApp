# Docker deployment

**For agents / operators deploying on a remote server:** follow
[`deploy/DEPLOY.md`](deploy/DEPLOY.md) on branch `docker-deployment`.

```bash
git clone -b docker-deployment --single-branch \
  https://github.com/SebKrantz/InfraRiskApp.git InfraRiskApp
cd InfraRiskApp
# Then open deploy/DEPLOY.md and execute every step.
```

Hazard GeoTIFFs are **not** in git. They must be downloaded to `data/rasters/`
on the server (see the deploy doc).
