# Download all hazard COGs listed in data/hazard_layers.csv into data/rasters/.
#
# This named vector MUST stay in sync with data/hazard_layers.csv (hazard → dataset_url).
# For Docker deployment on a remote server, prefer the Python script (stdlib only), which
# also rewrites CSV paths for the container:
#   python3 scripts/download_hazard_rasters.py --path-prefix /app/data/rasters
#
# Usage (from repo root, requires R):
#   Rscript data/download_rasters.R
#
# Behavior: creates data/rasters/ if needed; skips files that already exist; does NOT
# rewrite hazard_layers.csv (use the Python script for that after download).

file_names <- c(
  "Flood Hazard 25 Years - Existing climate" = "https://hazards-data.unepgrid.ch/global_pc_h25glob.tif",
  "Flood Hazard 50 Years - Existing climate" = "https://hazards-data.unepgrid.ch/global_pc_h50glob.tif",
  "Flood Hazard 100 Years - Existing climate" = "https://hazards-data.unepgrid.ch/global_pc_h100glob.tif",
  "Flood Hazard 25 Years - SSP1 Lower bound" = "https://hazards-data.unepgrid.ch/global_rcp26_h25glob.tif",
  "Flood Hazard 50 Years - SSP1 Lower bound" = "https://hazards-data.unepgrid.ch/global_rcp26_h50glob.tif",
  "Flood Hazard 100 Years - SSP1 Lower bound" = "https://hazards-data.unepgrid.ch/global_rcp26_h100glob.tif",
  "Flood Hazard 25 Years - SSP5 Upper bound" = "https://hazards-data.unepgrid.ch/global_rcp85_h25glob.tif",
  "Flood Hazard 50 Years - SSP5 Upper bound" = "https://hazards-data.unepgrid.ch/global_rcp85_h50glob.tif",
  "Flood Hazard 100 Years - SSP5 Upper bound" = "https://hazards-data.unepgrid.ch/global_rcp85_h100glob.tif",
  "Tropical Cyclone Wind - 25 Years" = "https://hazards-data.unepgrid.ch/Wind_T25.tif",
  "Tropical Cyclone Wind - 50 Years" = "https://hazards-data.unepgrid.ch/Wind_T50.tif",
  "Tropical Cyclone Wind - 100 Years" = "https://hazards-data.unepgrid.ch/Wind_T100.tif",
  "Tropical Cyclone Wind Climate Change - 25 Years" = "https://hazards-data.unepgrid.ch/Wind_CC_T25.tif",
  "Tropical Cyclone Wind Climate Change - 50 Years" = "https://hazards-data.unepgrid.ch/Wind_CC_T50.tif",
  "Tropical Cyclone Wind Climate Change - 100 Years" = "https://hazards-data.unepgrid.ch/Wind_CC_T100.tif",
  "Drought hazard SPI-6 5-year return period - Existing climate" = "https://hazards-data.unepgrid.ch/spi-06_past_sev3_GF.tif",
  "Drought hazard SPI-6 5-year return period - SSP1 Lower bound" = "https://hazards-data.unepgrid.ch/spi-06_SSP126_sev3_GF.tif",
  "Drought hazard SPI-6 5-year return period - SSP5 Upper bound" = "https://hazards-data.unepgrid.ch/spi-06_SSP585_sev3_GF.tif",
  "Average duration of a drought event (SPI-6) - Existing climate" = "https://hazards-data.unepgrid.ch/spi-06_past_dur_GF.tif",
  "Number of drought events in the analysed period (SPI-6) - Existing climate" = "https://hazards-data.unepgrid.ch/spi-06_past_num_GF.tif",
  "Susceptibility Class of Landslides Triggered By Precipitation - Existing climate" = "https://hazards-data.unepgrid.ch/n2_mosaic_wgs84_opt.tif",
  "Susceptibility Class of Landslides Triggered By Precipitation - Lower bound" = "https://hazards-data.unepgrid.ch/n3_mosaic_wgs84_opt.tif",
  "Susceptibility Class of Landslides Triggered By Precipitation - Upper bound" = "https://hazards-data.unepgrid.ch/n4_mosaic_wgs84_opt.tif",
  "Susceptibility Class of Landslides Triggered By Earthquakes" = "https://hazards-data.unepgrid.ch/n1_mosaic_wgs84_opt.tif",
  "Peak Ground Acceleration PGA - 250 Years" = "https://hazards-data.unepgrid.ch/PGA_250y.tif",
  "Peak Ground Acceleration PGA - 475 Years" = "https://hazards-data.unepgrid.ch/PGA_475y.tif",
  "Peak Ground Acceleration PGA - 975 Years" = "https://hazards-data.unepgrid.ch/PGA_975y.tif",
  "Building Exposure Model (BEM) - Total" = "https://hazards-data.unepgrid.ch/bem_5x5_valfis.tif",
  "Population distribution (GHSL) 2020" = "https://hazards-data.unepgrid.ch/GHS_POP_E2020_GLOBE_R2022A_54009_1000_V1_0_wgs84_opt.tif"
)

# Resolve paths relative to repo root (works whether cwd is repo root or data/)
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg)))
} else {
  normalizePath(".")
}
repo_root <- if (basename(script_dir) == "data") dirname(script_dir) else script_dir
download_dir <- file.path(repo_root, "data", "rasters")
dir.create(download_dir, recursive = TRUE, showWarnings = FALSE)

options(timeout = 3600)

cat("Downloading", length(file_names), "hazard rasters →", download_dir, "\n")

for (name in names(file_names)) {
  url <- file_names[[name]]
  file_name <- basename(url)
  dest <- file.path(download_dir, file_name)
  if (file.exists(dest)) {
    cat("SKIP (exists):", name, "→", file_name, "\n")
    next
  }
  cat("Downloading:", name, "\n")
  cat("  ", url, "\n")
  tryCatch(
    download.file(url, dest, mode = "wb", quiet = FALSE),
    error = function(e) {
      warning("FAILED: ", name, " — ", conditionMessage(e))
      if (file.exists(dest)) file.remove(dest)
    }
  )
}

cat("Done. Rasters are under:", download_dir, "\n")
cat("For Docker: next rewrite CSV paths with:\n")
cat("  python3 scripts/download_hazard_rasters.py --path-prefix /app/data/rasters\n")
