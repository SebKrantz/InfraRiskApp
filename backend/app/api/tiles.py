"""
Tile endpoints for serving COG tiles
"""
import io
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
import rasterio
from rasterio.windows import from_bounds
import numpy as np
from PIL import Image
import matplotlib.cm as cm

router = APIRouter()


def apply_colormap(data: np.ndarray, palette: str = 'viridis', vmin: float = None, vmax: float = None) -> np.ndarray:
    """Apply a colormap to raster data"""
    # Get colormap
    cmap = cm.get_cmap(palette)
    
    # Handle NaN and invalid values
    valid_mask = ~np.isnan(data) & (data >= 0.0) & (data < 1e15)
    all_valid = np.all(valid_mask)

    if all_valid:
        if vmax > vmin:
            normalized = (data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(data, dtype=np.float32)
    else:
        normalized = np.zeros_like(data, dtype=np.float32)
        if vmax > vmin:
            normalized[valid_mask] = (data[valid_mask] - vmin) / (vmax - vmin)
    
    # Apply colormap
    colored = cmap(normalized)
    
    # Convert to uint8 RGB
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return colored_uint8


@router.get("/tiles/{hazard_id}/{z}/{x}/{y}.png")
async def get_tile(
    hazard_id: str,
    z: int,
    x: int,
    y: int,
    palette: str = Query('viridis', description="Color palette"),
):
    """
    Get a tile from a COG hazard layer
    
    Parameters:
    - hazard_id: ID of hazard layer
    - z, x, y: Tile coordinates
    - palette: Color palette name
    """
    from app.api.hazards import load_hazards_dict, get_hazard_stats_cached
    
    try:
        # Get hazard info from cached dict
        hazards_dict = load_hazards_dict()
        
        if hazard_id not in hazards_dict:
            raise HTTPException(status_code=404, detail=f"Hazard layer '{hazard_id}' not found")
        
        hazard_data = hazards_dict[hazard_id]
        cog_url = hazard_data.get("dataset_url", "")
        
        if not cog_url:
            raise HTTPException(status_code=500, detail="Hazard layer missing dataset_url")
        
        # Get cached min/max values for consistent colormap scaling
        stats = get_hazard_stats_cached(hazard_id)
        if stats:
            vmin, vmax = stats
        else:
            vmin, vmax = 0.0, 1e6
        
        # Open COG
        with rasterio.open(cog_url) as src:
            # Calculate tile bounds in EPSG:4326 (geographic degrees)
            # All hazard rasters use EPSG:4326, so no CRS transformation needed
            import math
            n = 2.0 ** z
            lon_deg = (x / n) * 360.0 - 180.0
            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
            lat_deg = math.degrees(lat_rad)
            
            # Calculate bounds for this tile
            tile_size = 256
            pixel_size = 360.0 / n / tile_size
            
            minx = lon_deg
            maxx = lon_deg + pixel_size * tile_size
            miny = lat_deg - pixel_size * tile_size
            maxy = lat_deg
            
            # Bounds are already in EPSG:4326 (geographic degrees)
            bounds_4326 = (minx, miny, maxx, maxy)
            
            # Read data from COG
            try:
                window = from_bounds(*bounds_4326, src.transform)
                data = src.read(1, window=window, out_shape=(tile_size, tile_size))
                
                colored = apply_colormap(data, palette=palette, vmin=vmin, vmax=vmax)
                
                # Create PNG image
                img = Image.fromarray(colored, 'RGB')
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                return Response(content=img_buffer.read(), media_type="image/png")
            except Exception as e:
                print(f"Tile generation error: {e}")
                # Return transparent tile if read fails
                transparent = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
                img = Image.fromarray(transparent, 'RGBA')
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                return Response(content=img_buffer.read(), media_type="image/png")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating tile: {str(e)}")

