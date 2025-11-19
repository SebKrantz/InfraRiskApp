"""
Geospatial utility functions
"""

import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
from shapely.geometry import Point
from typing import Tuple, Optional
import pyogrio

def load_csv_points(file_path: str) -> gpd.GeoDataFrame:
    """
    Load a CSV file with point coordinates
    
    Supports various column name variations (case-agnostic):
    - lat/lon, lat/lng, latitude/longitude
    - y/x (for Y/X coordinate order)
    
    Supports both comma and semicolon delimiters (auto-detected).
    
    Args:
        file_path: Path to .csv file
        
    Returns:
        GeoDataFrame with Point geometries
    """
    # Read CSV - auto-detect delimiter and encoding
    # Try multiple encodings to handle different file formats
    import csv
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
    
    delimiter = ','
    encoding = 'utf-8'
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                first_line = f.readline()
                # Try to detect delimiter
                sniffer = csv.Sniffer()
                try:
                    delimiter = sniffer.sniff(first_line, delimiters=',;').delimiter
                except:
                    # Fallback: check if semicolon is present, otherwise use comma
                    delimiter = ';' if ';' in first_line else ','
                encoding = enc
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # Read with detected delimiter and encoding
    df = pd.read_csv(file_path, sep=delimiter, encoding=encoding)
    
    # Normalize column names to lowercase and strip whitespace
    original_columns = list(df.columns)
    df.columns = df.columns.str.lower().str.strip()
    
    # Find coordinate columns (case-agnostic matching)
    lat_col = None
    lon_col = None
    
    # Try different column name patterns (in priority order)
    # Priority: exact matches first, then check alternatives
    lat_patterns = ['lat', 'latitude', 'y']
    lon_patterns = ['lon', 'lng', 'longitude', 'long', 'x']
    
    # Check for latitude column (try patterns in order of preference)
    for pattern in lat_patterns:
        # Check for exact match (case-insensitive since we already normalized)
        if pattern in df.columns:
            lat_col = pattern
            break
    
    # Check for longitude column
    for pattern in lon_patterns:
        if pattern in df.columns:
            lon_col = pattern
            break
    
    # Special handling: if we found x and y but no lat/lon, 
    # assume y=latitude (north-south) and x=longitude (east-west)
    # This is the standard geographic convention
    
    if lat_col is None or lon_col is None:
        # Provide helpful error message with both original and normalized column names
        raise ValueError(
            f"Could not find coordinate columns in CSV. "
            f"Expected columns like: lat/lon, lat/lng, latitude/longitude, or y/x. "
            f"Found columns (original): {original_columns}. "
            f"Found columns (normalized): {list(df.columns)}"
        )
    
    # Extract coordinates
    lat = pd.to_numeric(df[lat_col], errors='coerce')
    lon = pd.to_numeric(df[lon_col], errors='coerce')
    
    # Check for missing values
    missing = lat.isna() | lon.isna()
    if missing.any():
        missing_count = missing.sum()
        total_count = len(df)
        print(f"Warning: {missing_count}/{total_count} rows have missing coordinates and will be dropped")
        lat = lat[~missing]
        lon = lon[~missing]
        df = df[~missing].copy()
    
    # Validate coordinate ranges
    if lat.min() < -90 or lat.max() > 90:
        raise ValueError(f"Latitude values out of range [-90, 90]. Found range: [{lat.min()}, {lat.max()}]")
    
    if lon.min() < -180 or lon.max() > 180:
        raise ValueError(f"Longitude values out of range [-180, 180]. Found range: [{lon.min()}, {lon.max()}]")
    
    # Create Point geometries
    geometry = [Point(lon_val, lat_val) for lon_val, lat_val in zip(lon, lat)]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    return gdf


def load_spatial_file(file_path: str) -> gpd.GeoDataFrame:
    """
    Load a spatial file (Shapefile or GeoPackage)
    
    Args:
        file_path: Path to .shp or .gpkg file
        
    Returns:
        GeoDataFrame
    """
    try:
        # Use pyogrio for faster I/O if available
        gdf = gpd.read_file(file_path, engine="pyogrio")
    except:
        # Fallback to fiona
        gdf = gpd.read_file(file_path)
    
    # Ensure CRS is set (if not, assume WGS84)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    
    return gdf


def validate_geometry_type(gdf: gpd.GeoDataFrame) -> str:
    """
    Validate that geometry type is Point or LineString
    
    Args:
        gdf: GeoDataFrame
        
    Returns:
        "Point" or "LineString"
    """
    geom_types = gdf.geometry.geom_type.unique()
    
    if len(geom_types) > 1:
        raise ValueError(f"Mixed geometry types found: {geom_types}. Only Point or LineString are supported.")
    
    geom_type = geom_types[0]
    
    if geom_type not in ["Point", "LineString", "MultiPoint", "MultiLineString"]:
        raise ValueError(f"Geometry type '{geom_type}' not supported. Only Point or LineString are allowed.")
    
    # Normalize to Point or LineString
    if geom_type in ["Point", "MultiPoint"]:
        return "Point"
    else:
        return "LineString"


def analyze_intersection(
    infrastructure_gdf: gpd.GeoDataFrame,
    hazard_raster_path: str,
    geometry_type: str,
    intensity_threshold: Optional[float] = None
) -> dict:
    """
    Analyze intersection between infrastructure and hazard raster
    
    All hazard rasters use EPSG:4326 (WGS84) as their CRS.
    
    Args:
        infrastructure_gdf: GeoDataFrame with infrastructure assets
        hazard_raster_path: Path or URL to hazard raster (assumed EPSG:4326)
        geometry_type: Geometry type ("Point" or "LineString") from stored metadata
        intensity_threshold: Optional threshold for filtering hazard intensity
        
    Returns:
        Dictionary with analysis results:
        - affected_count/unaffected_count (for points)
        - affected_meters/unaffected_meters (for lines)
        - affected_gdf: GeoDataFrame of affected features
    """
    # All hazard rasters use EPSG:4326
    # Transform infrastructure to EPSG:4326 if needed
    with rasterio.open(hazard_raster_path) as src:
        if infrastructure_gdf.crs != "EPSG:4326":
            infrastructure_gdf = infrastructure_gdf.to_crs("EPSG:4326")
                
        if len(infrastructure_gdf) == 0:
            return {
                "affected_count": 0,
                "unaffected_count": 0,
                "affected_meters": 0.0,
                "unaffected_meters": 0.0
            }
        
        # Sample raster values at infrastructure locations
        if geometry_type == "Point":
            # Sample points
            coords = [(geom.x, geom.y) for geom in infrastructure_gdf.geometry]
            raster_values = list(src.sample(coords))
            raster_values = [v[0] if len(v) > 0 and not np.isnan(v[0]) else None for v in raster_values]
            
            # Store exposure level (raster value at point location)
            infrastructure_gdf['exposure_level'] = [v if v is not None else None for v in raster_values]
            
            # Determine affected points
            if intensity_threshold is not None:
                affected_mask = [v is not None and v >= intensity_threshold for v in raster_values]
            else:
                affected_mask = [v is not None and v > 0 for v in raster_values]
            
            affected_count = sum(affected_mask)
            unaffected_count = len(infrastructure_gdf) - affected_count
            
            # Mark affected status in GeoDataFrame
            infrastructure_gdf['affected'] = affected_mask
            affected_gdf = infrastructure_gdf[affected_mask]
            
            # Clean NaN values from GeoDataFrame before returning
            # Replace NaN/Inf with None for JSON serialization
            import pandas as pd
            for col in infrastructure_gdf.columns:
                if col != 'geometry':
                    if infrastructure_gdf[col].dtype in ['float64', 'float32']:
                        infrastructure_gdf[col] = infrastructure_gdf[col].replace([np.nan, np.inf, -np.inf], [None, None, None])
                    elif pd.api.types.is_numeric_dtype(infrastructure_gdf[col]):
                        infrastructure_gdf[col] = infrastructure_gdf[col].replace([np.nan], [None])
            
            if len(affected_gdf) > 0:
                for col in affected_gdf.columns:
                    if col != 'geometry':
                        if affected_gdf[col].dtype in ['float64', 'float32']:
                            affected_gdf[col] = affected_gdf[col].replace([np.nan, np.inf, -np.inf], [None, None, None])
                        elif pd.api.types.is_numeric_dtype(affected_gdf[col]):
                            affected_gdf[col] = affected_gdf[col].replace([np.nan], [None])
            
            return {
                "affected_count": int(affected_count),
                "unaffected_count": int(unaffected_count),
                "affected_meters": 0.0,
                "unaffected_meters": 0.0,
                "affected_gdf": affected_gdf,
                "full_gdf": infrastructure_gdf  # Return full GDF with affected status
            }
        
        else:  # LineString
            # For lines, we need to sample along the line
            # This is a simplified approach - could be optimized
            affected_length = 0.0
            unaffected_length = 0.0
            affected_status = []
            exposure_levels_max = []
            exposure_levels_avg = []
            
            infrastructure_gdf = infrastructure_gdf.copy()
            
            for idx, row in infrastructure_gdf.iterrows():
                line = row.geometry
                length = line.length
                
                # Sample points along the line (every 100m or at least 5 points)
                num_samples = max(5, int(length / 100))
                distances = np.linspace(0, length, num_samples)
                
                sampled_points = [line.interpolate(d) for d in distances]
                coords = [(p.x, p.y) for p in sampled_points]
                
                try:
                    raster_values = list(src.sample(coords))
                    raster_values = [v[0] if len(v) > 0 and not np.isnan(v[0]) else None for v in raster_values]
                    
                    # Calculate exposure levels (max and avg from sampled points)
                    valid_values = [v for v in raster_values if v is not None]
                    if len(valid_values) > 0:
                        exposure_max = max(valid_values)
                        exposure_avg = sum(valid_values) / len(valid_values)
                    else:
                        exposure_max = None
                        exposure_avg = None
                    
                    exposure_levels_max.append(exposure_max)
                    exposure_levels_avg.append(exposure_avg)
                    
                    # Determine if line is affected (if any point above threshold)
                    if intensity_threshold is not None:
                        is_affected = any(v is not None and v >= intensity_threshold for v in raster_values)
                    else:
                        is_affected = any(v is not None and v > 0 for v in raster_values)
                    
                    affected_status.append(is_affected)
                    
                    if is_affected:
                        affected_length += length
                    else:
                        unaffected_length += length
                        
                except Exception:
                    # If sampling fails, assume unaffected
                    affected_status.append(False)
                    exposure_levels_max.append(None)
                    exposure_levels_avg.append(None)
                    unaffected_length += length
            
            # Mark affected status and exposure levels in GeoDataFrame
            infrastructure_gdf['affected'] = affected_status
            infrastructure_gdf['exposure_level_max'] = exposure_levels_max
            infrastructure_gdf['exposure_level_avg'] = exposure_levels_avg
            
            # Clean NaN values from GeoDataFrame before returning
            import pandas as pd
            for col in infrastructure_gdf.columns:
                if col != 'geometry':
                    if infrastructure_gdf[col].dtype in ['float64', 'float32']:
                        infrastructure_gdf[col] = infrastructure_gdf[col].replace([np.nan, np.inf, -np.inf], [None, None, None])
                    elif pd.api.types.is_numeric_dtype(infrastructure_gdf[col]):
                        infrastructure_gdf[col] = infrastructure_gdf[col].replace([np.nan], [None])
            
            # Ensure meters are valid floats (not NaN)
            affected_meters = float(affected_length) if not np.isnan(affected_length) else 0.0
            unaffected_meters = float(unaffected_length) if not np.isnan(unaffected_length) else 0.0
            
            return {
                "affected_count": 0,
                "unaffected_count": 0,
                "affected_meters": affected_meters,
                "unaffected_meters": unaffected_meters,
                "full_gdf": infrastructure_gdf  # Return full GDF with affected status
            }

