"""
Hazard layers endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import hashlib
import re
import csv

from app.config import settings

router = APIRouter()

# Module-level cache for parsed hazard data
_hazards_cache: Optional[Dict[str, Dict]] = None


def generate_id_from_name(name: str) -> str:
    """Generate a unique ID from hazard name by creating a slug"""
    # Convert to lowercase and replace spaces/special chars with underscores
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower().strip())
    # Remove leading/trailing underscores
    slug = slug.strip('_')
    # Use first 50 chars for ID
    return slug[:50] if slug else hashlib.md5(name.encode()).hexdigest()[:12]


def _read_hazard_csv(csv_path: Path) -> pd.DataFrame:
    """Reads hazard_layers.csv with robust delimiter and header detection."""
    try:
        # Try sniffing delimiter
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
            df = pd.read_csv(f, sep=dialect.delimiter)
    except (csv.Error, pd.errors.ParserError):
        # Fallback to semicolon if sniffing fails
        try:
            df = pd.read_csv(csv_path, sep=';')
        except pd.errors.ParserError:
            # Fallback to comma if semicolon fails
            df = pd.read_csv(csv_path, sep=',')
    
    # Normalize column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Map common column name variants to expected names
    column_mapping = {
        'url': 'dataset_url',
        'link': 'dataset_url',
        'background': 'background_paper',
        'metadata_url': 'background_paper'
    }
    df = df.rename(columns=column_mapping)
    
    return df


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
        
        # Validate required columns
        required_cols = ["hazard", "dataset_url"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"hazard_layers.csv missing required columns: {', '.join(missing_cols)}")
        
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
        
        # Read raster statistics
        try:
            with rasterio.open(cog_url) as src:
                # Read a sample to get statistics (for large rasters, use overviews or sampling)
                # For efficiency, we'll sample or use overviews if available
                data = src.read(1, out_shape=(1000, 1000)) if src.width > 1000 or src.height > 1000 else src.read(1)
                
                # Calculate statistics, ignoring NaN/NoData values
                valid_data = data[~np.isnan(data)]
                if len(valid_data) == 0:
                    return JSONResponse(content={
                        "min": 0,
                        "max": 0,
                        "mean": 0
                    })
                
                min_val = float(np.min(valid_data))
                max_val = float(np.max(valid_data))
                mean_val = float(np.mean(valid_data))
                
                return JSONResponse(content={
                    "min": min_val,
                    "max": max_val,
                    "mean": mean_val
                })
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error reading raster statistics: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting hazard statistics: {str(e)}"
        )

