"""
Export endpoints for generating high-resolution images
"""
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import numpy as np

from app.api.upload import uploaded_files
from app.api.hazards import load_hazards_dict
from app.api.analyze import get_cached_raster_values, get_cached_analysis_result
from app.utils.geospatial import analyze_intersection

router = APIRouter()

# Thread pool for blocking image generation
_export_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="export_worker")

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class ExportBarchartRequest(BaseModel):
    """Request model for barchart export"""
    file_id: str
    hazard_id: str
    intensity_threshold: Optional[float] = None


class ExportMapRequest(BaseModel):
    """Request model for map export"""
    file_id: str
    hazard_id: str
    color_palette: str = 'turbo'
    hazard_opacity: float = 0.6
    intensity_threshold: Optional[float] = None


def generate_barchart_png(
    analysis_result: dict,
    geometry_type: str,
    title: str,
    full_gdf: Optional[gpd.GeoDataFrame] = None
) -> bytes:
    """
    Generate PNG barchart from analysis results.
    
    Parameters:
    -----------
    analysis_result : dict
        Analysis result with summary statistics
    geometry_type : str
        'Point' or 'LineString'
    title : str
        Chart title
    full_gdf : GeoDataFrame, optional
        Full GeoDataFrame with vulnerability and damage_cost properties (for vulnerability mode)
    
    Returns:
    --------
    bytes : PNG image bytes
    """
    import pandas as pd
    
    # Check if vulnerability analysis data is available
    total_damage_cost = analysis_result.get('total_damage_cost')
    
    if total_damage_cost is not None:
        # Vulnerability analysis mode - show damage cost (left axis) and exposure/vulnerability (right axis)
        damage_cost = max(0, total_damage_cost)
        
        # Calculate exposure percentage
        exposure_pct = 0.0
        if geometry_type == 'Point':
            affected_count = analysis_result.get('affected_count', 0)
            total_features = analysis_result.get('total_features', 0)
            if total_features > 0:
                exposure_pct = (affected_count / total_features) * 100
        else:
            affected_meters = analysis_result.get('affected_meters', 0.0)
            unaffected_meters = analysis_result.get('unaffected_meters', 0.0)
            total_meters = affected_meters + unaffected_meters
            if total_meters > 0:
                exposure_pct = (affected_meters / total_meters) * 100
        
        # Calculate average vulnerability of exposed assets (as percentage)
        vulnerability_pct = 0.0
        if full_gdf is not None and 'vulnerability' in full_gdf.columns:
            if geometry_type == 'Point':
                # For points: simple average of vulnerability for affected points
                exposed = full_gdf[full_gdf['affected'] & full_gdf['vulnerability'].notna()]
                if len(exposed) > 0:
                    vulnerability_pct = exposed['vulnerability'].mean() * 100
            else:
                # For lines: length-weighted average vulnerability of exposed segments
                exposed = full_gdf[full_gdf['affected'] & full_gdf['vulnerability'].notna()]
                if len(exposed) > 0 and 'length_m' in exposed.columns:
                    # Calculate weighted average
                    exposed = exposed[exposed['length_m'] > 0]
                    if len(exposed) > 0:
                        weighted_sum = (exposed['vulnerability'] * exposed['length_m']).sum()
                        length_sum = exposed['length_m'].sum()
                        if length_sum > 0:
                            vulnerability_pct = (weighted_sum / length_sum) * 100
        
        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(6, 5))
        
        # Disable gridlines
        ax1.grid(False)
        
        # Left axis: damage cost
        categories = ['Damage Cost', 'Exposure', 'Vulnerability']
        damage_values = [damage_cost, 0, 0]
        pct_values = [0, exposure_pct, vulnerability_pct]
        
        x_pos = np.arange(len(categories))
        width = 0.6
        
        # Plot damage cost on left axis
        bars1 = ax1.bar(x_pos[0], damage_cost, width, color='#f59e0b', alpha=0.8)
        ax1.set_ylabel('Damage Cost (USD)', fontsize=12, color='#4b5563')
        ax1.tick_params(axis='y', labelcolor='#4b5563')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['Damage', 'Exposure', 'Vulnerability'], fontsize=11, color='#4b5563')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        # Set y-axis limits to add margin above damage cost bar
        ax1.set_ylim(bottom=0, top=damage_cost * 1.15)
        
        # Format left axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Right axis: percentages
        ax2 = ax1.twinx()
        ax2.grid(False)  # Disable gridlines on right axis too
        bars2 = ax2.bar(x_pos[1], exposure_pct, width, color='#3b82f6', alpha=0.8)
        bars3 = ax2.bar(x_pos[2], vulnerability_pct, width, color='#10b981', alpha=0.8)
        ax2.set_ylabel('Percentage (%)', fontsize=12, color='#4b5563')
        ax2.tick_params(axis='y', labelcolor='#4b5563')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        # Damage cost
        ax1.text(x_pos[0], damage_cost + max(damage_cost, 1) * 0.01, 
                f'${damage_cost:,.0f}', ha='center', va='bottom', fontsize=10, color='#f59e0b')
        # Exposure
        ax2.text(x_pos[1], exposure_pct + 1, 
                f'{exposure_pct:.1f}%', ha='center', va='bottom', fontsize=10, color='#3b82f6')
        # Vulnerability
        ax2.text(x_pos[2], vulnerability_pct + 1, 
                f'{vulnerability_pct:.1f}%', ha='center', va='bottom', fontsize=10, color='#10b981')
        
        plt.tight_layout()
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return img_buffer.getvalue()
    else:
        # Standard exposure analysis mode
        if geometry_type == 'Point':
            affected = analysis_result.get('affected_count', 0)
            unaffected = analysis_result.get('unaffected_count', 0)
            ylabel = 'Count'
        else:
            affected = analysis_result.get('affected_meters', 0)
            unaffected = analysis_result.get('unaffected_meters', 0)
            ylabel = 'Length (meters)'
        
        plot_data = pd.DataFrame({
            'Category': ['Affected', 'Unaffected'],
            'Value': [affected, unaffected]
        })
        
        palette = ['#d62728', '#2ca02c']  # Red for affected, green for unaffected
        
        # Create plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(data=plot_data, x='Category', y='Value', 
                    palette=palette, ax=ax)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        max_value = max(plot_data['Value'])
        
        # Set y-axis limits to add margin above bars
        ax.set_ylim(bottom=0, top=max_value * 1.15)
        
        for i, v in enumerate(plot_data['Value']):
            label = f'{v:,.0f}'
            ax.text(i, v + max_value * 0.01, 
                    label, ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='PNG', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return img_buffer.getvalue()


def generate_map_png(
    infrastructure_gdf: gpd.GeoDataFrame,
    hazard_raster_path: str,
    geometry_type: str,
    title: str,
    color_palette: str = 'turbo',
    hazard_opacity: float = 0.6
) -> bytes:
    """
    Generate PNG map from infrastructure and hazard data.
    
    Parameters:
    -----------
    infrastructure_gdf : GeoDataFrame
        Infrastructure with 'affected' column (in WGS84)
    hazard_raster_path : str
        Path/URL to hazard raster
    geometry_type : str
        'Point' or 'LineString'
    title : str
        Map title
    color_palette : str
        Color palette name for hazard visualization
    hazard_opacity : float
        Opacity for hazard layer (0-1)
    
    Returns:
    --------
    bytes : PNG image bytes
    """
    if infrastructure_gdf is None or len(infrastructure_gdf) == 0:
        raise ValueError("No infrastructure data to plot")
    
    # Ensure infrastructure is in WGS84
    if infrastructure_gdf.crs != 'EPSG:4326':
        infrastructure_gdf = infrastructure_gdf.to_crs('EPSG:4326')
    
    # Get bounding box from infrastructure
    bounds = infrastructure_gdf.total_bounds
    margin = (bounds[2] - bounds[0]) * 0.1
    bbox = [bounds[0] - margin, bounds[1] - margin, 
            bounds[2] + margin, bounds[3] + margin]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Load and clip hazard raster using rasterio (much faster than xarray)
    try:
        import rasterio
        from rasterio.windows import from_bounds, bounds as window_bounds
        
        with rasterio.open(hazard_raster_path) as src:
            # Create window from bounding box
            window = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform)
            
            # Downsample to reasonable resolution for visualization (max 2000x2000)
            # Calculate target resolution based on window size
            window_height = window.height
            window_width = window.width
            
            # Target max dimension
            max_dim = 2000
            if window_height > max_dim or window_width > max_dim:
                scale = max(window_height, window_width) / max_dim
                out_shape = (int(window_height / scale), int(window_width / scale))
            else:
                out_shape = (int(window_height), int(window_width))
            
            # Read only the windowed, downsampled data
            hazard_data = src.read(1, window=window, out_shape=out_shape, boundless=True, fill_value=np.nan)
            
            # Get transform for the downsampled window
            window_transform = src.window_transform(window)
            
            # Calculate extent from window bounds (more reliable than transform calculation)
            # Get the actual bounds of the window
            win_bounds = window_bounds(window, src.transform)
            extent = [
                win_bounds[0],  # minx
                win_bounds[2],  # maxx
                win_bounds[1],  # miny
                win_bounds[3]   # maxy
            ]
            
            # Apply colormap
            import matplotlib.cm as cm
            cmap = cm.get_cmap(color_palette)
            
            # Normalize data
            valid_mask = ~np.isnan(hazard_data) & (hazard_data >= 0.0) & (hazard_data < 1e15)
            if np.any(valid_mask):
                vmin = np.nanmin(hazard_data[valid_mask])
                vmax = np.nanmax(hazard_data[valid_mask])
                
                if vmax > vmin:
                    # Use sqrt normalization like tile service
                    normalized = (np.sqrt(hazard_data) - np.sqrt(vmin)) / (np.sqrt(vmax) - np.sqrt(vmin))
                    normalized = np.clip(normalized, 0, 1)
                else:
                    normalized = np.zeros_like(hazard_data, dtype=np.float32)
                normalized[~valid_mask] = np.nan
                
                # Apply colormap
                colored = cmap(normalized)
                colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
                
                # Plot hazard raster
                im = ax.imshow(colored_uint8, extent=extent, 
                              alpha=hazard_opacity, interpolation='bilinear', 
                              aspect='auto', origin='upper', zorder=1)
                
                # Create a ScalarMappable for the colorbar
                # We need to create a normalization that matches our sqrt transformation
                from matplotlib.colors import Normalize
                from matplotlib.cm import ScalarMappable
                
                # Create a custom normalization class for sqrt scaling
                class SqrtNormalize(Normalize):
                    def __init__(self, vmin=None, vmax=None, clip=False):
                        super().__init__(vmin, vmax, clip)
                    
                    def __call__(self, value, clip=None):
                        # Apply sqrt normalization
                        result = np.ma.masked_array(value, mask=np.ma.getmask(value))
                        if self.vmin is not None and self.vmax is not None and self.vmax > self.vmin:
                            sqrt_vmin = np.sqrt(self.vmin)
                            sqrt_vmax = np.sqrt(self.vmax)
                            result = (np.sqrt(result) - sqrt_vmin) / (sqrt_vmax - sqrt_vmin)
                            result = np.clip(result, 0, 1)
                        return result
                
                # Create ScalarMappable for colorbar
                norm = SqrtNormalize(vmin=vmin, vmax=vmax)
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                
                # Add colorbar
                cbar = plt.colorbar(sm, ax=ax, orientation='vertical', 
                                   pad=0.02, shrink=0.6, aspect=20)
                cbar.set_label('Hazard Intensity', rotation=270, labelpad=15, fontsize=10)
                cbar.ax.tick_params(labelsize=9)
    except Exception as e:
        print(f"Warning: Could not load hazard raster: {e}")
        # Continue without hazard layer
    
    # Plot infrastructure
    if 'affected' in infrastructure_gdf.columns:
        affected_gdf = infrastructure_gdf[infrastructure_gdf['affected']]
        unaffected_gdf = infrastructure_gdf[~infrastructure_gdf['affected']]
        
        if geometry_type == 'Point':
            if len(unaffected_gdf) > 0:
                unaffected_gdf.plot(ax=ax, color='#2ca02c', markersize=50, 
                                  marker='o', label='Unaffected', alpha=0.8, 
                                  edgecolor='black', linewidth=0.5, zorder=3)
            if len(affected_gdf) > 0:
                affected_gdf.plot(ax=ax, color='#d62728', markersize=50, 
                                marker='o', label='Affected', alpha=0.8, 
                                edgecolor='black', linewidth=0.5, zorder=3)
        else:
            if len(unaffected_gdf) > 0:
                unaffected_gdf.plot(ax=ax, color='#2ca02c', linewidth=2, 
                                  label='Unaffected', alpha=0.8, zorder=3)
            if len(affected_gdf) > 0:
                affected_gdf.plot(ax=ax, color='#d62728', linewidth=2, 
                               label='Affected', alpha=0.8, zorder=3)
    else:
        # No analysis - plot all in gray
        if geometry_type == 'Point':
            infrastructure_gdf.plot(ax=ax, color='#6b7280', markersize=50, 
                                  marker='o', alpha=0.8, 
                                  edgecolor='black', linewidth=0.5, zorder=3)
        else:
            infrastructure_gdf.plot(ax=ax, color='#6b7280', linewidth=2, 
                                  alpha=0.8, zorder=3)
    
    # Set extent
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    
    # Add legend if we have affected/unaffected
    if 'affected' in infrastructure_gdf.columns:
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.set_axis_off()
    plt.tight_layout()
    
    # Save to bytes buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='PNG', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return img_buffer.getvalue()


@router.post("/export/barchart")
async def export_barchart(request: ExportBarchartRequest):
    """
    Export analysis results as a high-resolution PNG barchart.
    """
    # Check if file exists
    if request.file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    
    file_info = uploaded_files[request.file_id]
    geometry_type = file_info["geometry_type"]
    infrastructure_gdf = file_info["gdf"]
    
    # Get hazard info
    hazards_dict = load_hazards_dict()
    if request.hazard_id not in hazards_dict:
        raise HTTPException(status_code=404, detail=f"Hazard layer '{request.hazard_id}' not found")
    
    hazard_data = hazards_dict[request.hazard_id]
    hazard_url = hazard_data.get("dataset_url", "")
    
    if not hazard_url:
        raise HTTPException(status_code=500, detail="Hazard layer missing dataset_url")
    
    try:
        # Since button only appears after analysis completes, we should always have a cached result
        # Try to find cached result (check with provided threshold first, then try None)
        cached_analysis = get_cached_analysis_result(
            request.file_id, 
            request.hazard_id, 
            request.intensity_threshold
        )
        
        # If not found with threshold, try without threshold (for cases where threshold wasn't used)
        if cached_analysis is None:
            cached_analysis = get_cached_analysis_result(
                request.file_id, 
                request.hazard_id, 
                None
            )
        
        if cached_analysis is None:
            raise HTTPException(
                status_code=404,
                detail="No analysis results found. Please run analysis first."
            )
        
        # Use cached result - no recalculation needed!
        analysis_result = cached_analysis
        
        # Build summary for barchart (include vulnerability data if available)
        summary = {
            "affected_count": analysis_result.get("affected_count", 0),
            "unaffected_count": analysis_result.get("unaffected_count", 0),
            "affected_meters": analysis_result.get("affected_meters", 0.0),
            "unaffected_meters": analysis_result.get("unaffected_meters", 0.0),
            "total_damage_cost": analysis_result.get("total_damage_cost"),
            "total_features": file_info["feature_count"]
        }
        
        # Get full_gdf for vulnerability calculations if available
        full_gdf = None
        if "full_gdf" in analysis_result:
            full_gdf = analysis_result["full_gdf"]
        
        # Generate title
        hazard_name = hazard_data.get("hazard", request.hazard_id)
        if summary.get("total_damage_cost") is not None:
            title = f"{hazard_name} - Vulnerability Analysis"
        else:
            title = f"{hazard_name} - Infrastructure Exposure Analysis"
        
        # Generate PNG
        loop = asyncio.get_event_loop()
        png_bytes = await loop.run_in_executor(
            _export_executor,
            generate_barchart_png,
            summary,
            geometry_type,
            title,
            full_gdf
        )
        
        # Generate filename
        filename = f"barchart_{request.file_id[:8]}_{request.hazard_id[:8]}.png"
        
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating barchart: {str(e)}"
        )


@router.post("/export/map")
async def export_map(request: ExportMapRequest):
    """
    Export analysis results as a high-resolution PNG map.
    """
    # Check if file exists
    if request.file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    
    file_info = uploaded_files[request.file_id]
    geometry_type = file_info["geometry_type"]
    infrastructure_gdf = file_info["gdf"]
    
    # Get hazard info
    hazards_dict = load_hazards_dict()
    if request.hazard_id not in hazards_dict:
        raise HTTPException(status_code=404, detail=f"Hazard layer '{request.hazard_id}' not found")
    
    hazard_data = hazards_dict[request.hazard_id]
    hazard_url = hazard_data.get("dataset_url", "")
    
    if not hazard_url:
        raise HTTPException(status_code=500, detail="Hazard layer missing dataset_url")
    
    try:
        # Since button only appears after analysis completes, we should always have a cached result
        # Try to find cached result (check with provided threshold first, then try None)
        cached_analysis = get_cached_analysis_result(
            request.file_id, 
            request.hazard_id, 
            request.intensity_threshold
        )
        
        # If not found with threshold, try without threshold (for cases where threshold wasn't used)
        if cached_analysis is None:
            cached_analysis = get_cached_analysis_result(
                request.file_id, 
                request.hazard_id, 
                None
            )
        
        if cached_analysis is None:
            raise HTTPException(
                status_code=404,
                detail="No analysis results found. Please run analysis first."
            )
        
        # Use cached result - no recalculation needed!
        analysis_result = cached_analysis
        
        # Get infrastructure GeoDataFrame with affected status
        if "full_gdf" in analysis_result:
            display_gdf = analysis_result["full_gdf"]
        else:
            display_gdf = infrastructure_gdf.copy()
            display_gdf['affected'] = False
        
        # Generate title
        hazard_name = hazard_data.get("hazard", request.hazard_id)
        title = f"{hazard_name} - Infrastructure Exposure Map"
        
        # Generate PNG
        loop = asyncio.get_event_loop()
        png_bytes = await loop.run_in_executor(
            _export_executor,
            generate_map_png,
            display_gdf,
            hazard_url,
            geometry_type,
            title,
            request.color_palette,
            request.hazard_opacity
        )
        
        # Generate filename
        filename = f"map_{request.file_id[:8]}_{request.hazard_id[:8]}.png"
        
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating map: {str(e)}"
        )

