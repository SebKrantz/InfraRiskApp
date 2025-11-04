"""
Tile endpoints for serving COG tiles
"""
import io
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import numpy as np
from PIL import Image
import matplotlib.cm as cm

router = APIRouter()


def apply_colormap(data: np.ndarray, palette: str = 'viridis', vmin: float = None, vmax: float = None) -> np.ndarray:
    """Apply a colormap to raster data"""
    # Get colormap
    cmap = cm.get_cmap(palette)
    
    # Normalize data
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data) + 1
    
    # Handle NaN and invalid values
    valid_mask = ~np.isnan(data) & (data >= vmin) & (data < vmax)
    
    # Normalize to [0, 1]
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
    from app.api.hazards import load_hazards_dict
    
    try:
        # Get hazard info from cached dict
        hazards_dict = load_hazards_dict()
        
        if hazard_id not in hazards_dict:
            raise HTTPException(status_code=404, detail=f"Hazard layer '{hazard_id}' not found")
        
        hazard_data = hazards_dict[hazard_id]
        cog_url = hazard_data.get("dataset_url", "")
        
        if not cog_url:
            raise HTTPException(status_code=500, detail="Hazard layer missing dataset_url")
        
        # Open COG
        with rasterio.open(cog_url) as src:
            # Calculate tile bounds in Web Mercator (EPSG:3857)
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
            
            # Transform bounds to raster CRS
            bounds_3857 = (minx, miny, maxx, maxy)
            bounds_raster = transform_bounds('EPSG:3857', src.crs, *bounds_3857)
            
            # Read data from COG
            try:
                window = from_bounds(*bounds_raster, src.transform)
                data = src.read(1, window=window, out_shape=(tile_size, tile_size))
                
                # Apply colormap
                colored = apply_colormap(data, palette=palette, vmin=0, vmax=np.inf)
                
                # Create PNG image
                img = Image.fromarray(colored, 'RGB')
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                return Response(content=img_buffer.read(), media_type="image/png")
            except Exception as e:
                # Return transparent tile if read fails
                transparent = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
                img = Image.fromarray(transparent, 'RGBA')
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                return Response(content=img_buffer.read(), media_type="image/png")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating tile: {str(e)}")

