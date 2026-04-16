"""
Hazard layers endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import hashlib
import re
import csv

from app.config import settings

router = APIRouter()

# Module-level cache for parsed hazard data
_hazards_cache: Optional[Dict[str, Dict]] = None

# Module-level cache for hazard statistics (min, max) as tuples
_hazard_stats_cache: Dict[str, Tuple[float, float]] = {}


def generate_id_from_name(name: str) -> str:
    """Generate a unique ID from hazard name by creating a slug"""
    # Convert to lowercase and replace spaces/special chars with underscores
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower().strip())
    # Remove leading/trailing underscores
    slug = slug.strip('_')
    # Use first 50 chars for ID
    return slug[:50] if slug else hashlib.md5(name.encode()).hexdigest()[:12]


def _read_hazard_csv(csv_path: Path) -> pd.DataFrame:
    """
    Reads hazard_layers.csv with delimiter and encoding detection.

    Project convention is semicolon-separated (GIRI-style). csv.Sniffer() often
    mis-detects (e.g. '_' from URLs/text), so we try explicit separators and
    encodings before falling back to sniffing.
    """
    column_mapping = {
        'url': 'dataset_url',
        'link': 'dataset_url',
        'background': 'background_paper',
        'metadata_url': 'background_paper',
    }

    def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        # Excel/UTF-8 BOM and stray zero-width chars on header names
        raw.columns = (
            raw.columns.astype(str)
            .str.strip()
            .str.replace('\ufeff', '', regex=False)
            .str.replace('\u200b', '', regex=False)
            .str.lower()
        )
        return raw.rename(columns=column_mapping)

    required = {'hazard', 'dataset_url'}
    encodings = ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252')
    separators = (';', ',', '\t')

    for encoding in encodings:
        for sep in separators:
            try:
                df = _normalize(
                    pd.read_csv(csv_path, sep=sep, encoding=encoding, engine='python')
                )
                if required.issubset(df.columns):
                    return df
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
                continue

    # Last resort: sniff delimiter — validate columns; Sniffer is unreliable on messy rows
    for encoding in encodings:
        try:
            with open(csv_path, 'r', newline='', encoding=encoding) as f:
                sample = f.read(8192)
                dialect = csv.Sniffer().sniff(sample)
                df = _normalize(
                    pd.read_csv(csv_path, sep=dialect.delimiter, encoding=encoding, engine='python')
                )
                if required.issubset(df.columns):
                    return df
        except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
            continue

    # Help callers debug: show what the header line looks like with default encoding
    try:
        first = csv_path.read_text(encoding='utf-8-sig', errors='replace').splitlines()[:3]
        preview = ' | '.join(line[:120] for line in first if line.strip())
    except OSError:
        preview = '(could not read file preview)'

    raise ValueError(
        'Could not parse hazard_layers.csv: no encoding/separator pair produced columns '
        '"hazard" and "dataset_url". Use a header row with those names (semicolon- or comma-separated). '
        f'File preview: {preview!r}'
    )


def load_hazards_dict() -> Dict[str, Dict]:
    """
    Load hazard_layers.csv and return as Dict[str, Dict] where:
    - Outer dict keys are hazard_id (generated from hazard name)
    - Inner dict contains CSV row data with column names as keys
    
    Caches the result in module-level _hazards_cache.
    """
    global _hazards_cache
    
    # Return cached data if available
    if _hazards_cache is not None:
        return _hazards_cache
    
    csv_path = settings.HAZARD_LAYERS_CSV
    
    if not csv_path.exists():
        _hazards_cache = {}
        return _hazards_cache
    
    try:
        # Read CSV with robust delimiter detection
        df = _read_hazard_csv(csv_path)
        
        # Validate required columns (belt-and-suspenders after _read_hazard_csv)
        required_cols = ["hazard", "dataset_url"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            found = ", ".join(str(c) for c in df.columns)
            raise ValueError(
                f"hazard_layers.csv missing required columns: {', '.join(missing_cols)}. "
                f"Found columns: [{found}]. Expected a header with hazard and dataset_url."
            )
        
        # Build dict of dicts keyed by hazard_id
        hazards_dict = {}
        for _, row in df.iterrows():
            hazard_name = str(row["hazard"]).strip()
            hazard_id = generate_id_from_name(hazard_name)
            
            # Create inner dict with all CSV columns as keys
            hazard_data = {}
            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    hazard_data[col] = str(value).strip()
                else:
                    hazard_data[col] = None
            
            # Ensure required fields are present
            hazard_data["hazard_id"] = hazard_id  # Add generated ID for convenience
            
            hazards_dict[hazard_id] = hazard_data
        
        # Cache the result
        _hazards_cache = hazards_dict
        return hazards_dict
        
    except pd.errors.EmptyDataError:
        _hazards_cache = {}
        return _hazards_cache
    except Exception as e:
        raise ValueError(f"Error reading hazard_layers.csv: {str(e)}")


def clear_hazards_cache():
    """Clear the hazards cache (useful for testing or when CSV is updated)"""
    global _hazards_cache
    _hazards_cache = None


def get_hazard_stats_cached(hazard_id: str) -> Optional[Tuple[float, float]]:
    """
    Get cached hazard statistics (min, max) for a hazard layer.
    Returns None if stats are not cached yet, otherwise returns (min, max) tuple.
    """
    return _hazard_stats_cache.get(hazard_id)


@router.get("/hazards")
async def get_hazards():
    """
    Get list of available hazard layers from hazard_layers.csv
    
    Expected CSV format (semicolon-delimited):
    - hazard: display name (used to generate id)
    - dataset_url: URL or path to COG GeoTIFF
    - description: optional description
    - background_paper: optional metadata/background paper URL or path
    """
    try:
        hazards_dict = load_hazards_dict()
        
        # Convert to list format for API response
        hazards = []
        for hazard_id, hazard_data in hazards_dict.items():
            hazard = {
                "id": hazard_id,
                "name": hazard_data.get("hazard", ""),
                "url": hazard_data.get("dataset_url", "")
            }
            
            # Add optional fields if present
            if hazard_data.get("description"):
                hazard["description"] = hazard_data["description"]
            
            if hazard_data.get("background_paper"):
                hazard["metadata"] = hazard_data["background_paper"]

            # Surface unit from CSV if present (used by frontend for labels)
            if hazard_data.get("unit"):
                hazard["unit"] = hazard_data["unit"]

            if hazard_data.get("category"):
                hazard["category"] = hazard_data["category"]
            
            hazards.append(hazard)
        
        return JSONResponse(content={
            "hazards": hazards,
            "count": len(hazards)
        })
        
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading hazard_layers.csv: {str(e)}"
        )


@router.get("/hazards/{hazard_id}")
async def get_hazard_info(hazard_id: str):
    """Get detailed information about a specific hazard layer"""
    try:
        hazards_dict = load_hazards_dict()
        
        if hazard_id not in hazards_dict:
            raise HTTPException(status_code=404, detail=f"Hazard layer '{hazard_id}' not found")
        
        hazard_data = hazards_dict[hazard_id]
        
        hazard_info = {
            "id": hazard_id,
            "name": hazard_data.get("hazard", ""),
            "url": hazard_data.get("dataset_url", "")
        }
        
        # Add optional fields
        if hazard_data.get("description"):
            hazard_info["description"] = hazard_data["description"]
        
        if hazard_data.get("background_paper"):
            hazard_info["metadata"] = hazard_data["background_paper"]

        # Include unit field for detailed hazard info as well
        if hazard_data.get("unit"):
            hazard_info["unit"] = hazard_data["unit"]

        if hazard_data.get("category"):
            hazard_info["category"] = hazard_data["category"]
        
        return JSONResponse(content=hazard_info)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading hazard layer: {str(e)}"
        )


@router.get("/hazards/{hazard_id}/stats")
async def get_hazard_stats(hazard_id: str):
    """Get statistics (min, max) for a hazard raster"""
    import rasterio
    import numpy as np
    
    try:
        hazards_dict = load_hazards_dict()
        
        if hazard_id not in hazards_dict:
            raise HTTPException(status_code=404, detail=f"Hazard layer '{hazard_id}' not found")
        
        hazard_data = hazards_dict[hazard_id]
        cog_url = hazard_data.get("dataset_url", "")
        
        if not cog_url:
            raise HTTPException(status_code=500, detail="Hazard layer missing dataset_url")
        
        # Read raster statistics; try overview first (fewer tiles), then fallback so broken TIFFs don't 500
        def _stats_from_data(data: np.ndarray) -> Optional[Tuple[float, float]]:
            valid_mask = ~np.isnan(data) & (data >= 0.0) & (data < 1e15)
            valid_data = data if np.all(valid_mask) else data[valid_mask]
            if len(valid_data) == 0:
                return None
            return float(np.min(valid_data)), float(np.max(valid_data))

        last_error: Optional[Exception] = None
        try:
            with rasterio.open(cog_url) as src:
                # 1) Try reading from smallest overview (largest decimation) to avoid bad tiles
                ov = src.overviews(1)
                if ov:
                    decim = max(ov)
                    h, w = src.height // decim, src.width // decim
                    if h > 0 and w > 0:
                        try:
                            data = src.read(1, out_shape=(h, w))
                            stats = _stats_from_data(data)
                            if stats is not None:
                                min_val, max_val = stats
                                _hazard_stats_cache[hazard_id] = (np.sqrt(min_val), np.sqrt(max_val))
                                return JSONResponse(content={"min": min_val, "max": max_val})
                        except Exception as e:
                            last_error = e
                # 2) Try full-resolution sample (1000x1000 or full)
                try:
                    data = src.read(1, out_shape=(1000, 1000)) if (src.width > 1000 or src.height > 1000) else src.read(1)
                    stats = _stats_from_data(data)
                    if stats is not None:
                        min_val, max_val = stats
                        _hazard_stats_cache[hazard_id] = (np.sqrt(min_val), np.sqrt(max_val))
                        return JSONResponse(content={"min": min_val, "max": max_val})
                except Exception as e:
                    last_error = e
        except Exception as e:
            # open() or directory read failed (e.g. corrupt TIFF directory)
            last_error = e
        # 3) Fallback so UI and tiles still load (colormap scale may be wrong)
        _hazard_stats_cache[hazard_id] = (0.0, 1.0)
        if last_error is not None:
            import logging
            logging.getLogger(__name__).warning(
                "Raster stats read failed, using fallback min=0 max=1: %s",
                last_error,
                exc_info=True,
            )
        return JSONResponse(content={"min": 0, "max": 1})
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting hazard statistics: {str(e)}"
        )

