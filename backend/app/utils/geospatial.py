"""
Geospatial utility functions
"""

import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
from typing import Tuple, Optional
import pyogrio
from pyproj import Geod

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
        df = df[~missing]
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
        - full_gdf: GeoDataFrame with all features and affected status
    """
    # All hazard rasters use EPSG:4326
    # Infrastructure GeoDataFrame is already in WGS84 (EPSG:4326) from upload
    with rasterio.open(hazard_raster_path) as src:
                
        if len(infrastructure_gdf) == 0:
            return {
                "affected_count": 0,
                "unaffected_count": 0,
                "affected_meters": 0.0,
                "unaffected_meters": 0.0
            }
        
        # Sample raster values at infrastructure locations
        if geometry_type == "Point":
            # Extract coordinates as numpy arrays (vectorized, much faster than list comprehension)
            x_coords = infrastructure_gdf.geometry.x.to_numpy()
            y_coords = infrastructure_gdf.geometry.y.to_numpy()
            coords = np.column_stack([x_coords, y_coords])
            
            # Sample all points at once and convert to numpy array
            raster_values = np.array([v[0] if len(v) > 0 else np.nan for v in src.sample(coords)])
            
            # Vectorized operations for valid/affected status (much faster than Python loops)
            valid_mask = ~np.isnan(raster_values)
            if intensity_threshold is not None:
                affected_mask = valid_mask & (raster_values >= intensity_threshold)
            else:
                affected_mask = valid_mask & (raster_values > 0)
            
            # Store exposure level - convert NaN to None for JSON serialization
            exposure_values = np.where(valid_mask, raster_values, np.nan)
            infrastructure_gdf['exposure_level'] = exposure_values
            infrastructure_gdf['exposure_level'] = infrastructure_gdf['exposure_level'].replace({np.nan: None})
            
            # Mark affected status in GeoDataFrame
            infrastructure_gdf['affected'] = affected_mask
            
            affected_count = int(affected_mask.sum())
            unaffected_count = len(infrastructure_gdf) - affected_count
            
            return {
                "affected_count": affected_count,
                "unaffected_count": unaffected_count,
                "affected_meters": 0.0,
                "unaffected_meters": 0.0,
                "full_gdf": infrastructure_gdf  # Return full GDF with affected status
            }
        
        else:  # LineString
            # Sample lines at 100m intervals and split into affected/non-affected segments
            # Using geodesic distance for accurate sampling (matching notebook implementation)
            affected_length = 0.0
            unaffected_length = 0.0
            segment_rows = []
            geod = Geod(ellps="WGS84")
            
            for idx, row in infrastructure_gdf.iterrows():
                line = row.geometry
                
                # Handle MultiLineString (matching notebook approach)
                if line.geom_type == 'MultiLineString':
                    lines = list(line.geoms)
                else:
                    lines = [line]
                
                # Process each line separately
                for single_line in lines:
                    line_coords = list(single_line.coords)
                    
                    if len(line_coords) < 2:
                        continue
                    
                    # Calculate total geodesic length in meters
                    total_length_m = geod.line_length(*zip(*line_coords))
                    
                    if total_length_m == 0:
                        # Zero-length line, skip
                        continue
                    
                    # Sample points along the line at 100m intervals (matching notebook)
                    interval_meters = 100.0
                    sampled_points = []
                    
                    # Always include start point
                    sampled_points.append(line_coords[0])
                    
                    # Sample at regular intervals
                    current_distance = 0.0
                    while current_distance < total_length_m:
                        current_distance += interval_meters
                        if current_distance >= total_length_m:
                            # Include end point
                            sampled_points.append(line_coords[-1])
                            break
                        
                        # Interpolate point at this distance using normalized interpolation
                        normalized_dist = current_distance / total_length_m
                        point = single_line.interpolate(normalized_dist, normalized=True)
                        sampled_points.append((point.x, point.y))
                    
                    # Ensure end point is included
                    if sampled_points[-1] != line_coords[-1]:
                        sampled_points.append(line_coords[-1])
                    
                    if len(sampled_points) < 2:
                        continue
                    
                    try:
                        # Get raster values at sampled points
                        coords = [(x, y) for x, y in sampled_points]
                        raster_values = list(src.sample(coords))
                        # Extract first band value, converting NaN to None for JSON serialization
                        raster_values = [None if len(v) == 0 or np.isnan(v[0]) else float(v[0]) for v in raster_values]
                        
                        # Classify each point as affected or not (matching notebook approach)
                        if intensity_threshold is not None:
                            point_affected = [v is not None and v >= intensity_threshold for v in raster_values]
                        else:
                            point_affected = [v is not None and v > 0 for v in raster_values]
                        
                        # Create segments: group consecutive points with same affected status
                        # (matching notebook approach)
                        current_segment_points = [sampled_points[0]]
                        current_segment_indices = [0]  # Track which sampled points are in this segment
                        current_affected = point_affected[0]
                        
                        for i in range(1, len(sampled_points)):
                            if point_affected[i] == current_affected:
                                # Same status, continue current segment
                                current_segment_points.append(sampled_points[i])
                                current_segment_indices.append(i)
                            else:
                                # Status changed, create segment from current points
                                if len(current_segment_points) >= 2:
                                    segment_geom = LineString(current_segment_points)
                                    segment_length_m = geod.line_length(*zip(*current_segment_points))
                                    
                                    segment_values = [raster_values[j] for j in current_segment_indices if raster_values[j] is not None]
                                    if len(segment_values) > 0:
                                        avg_value = sum(segment_values) / len(segment_values)
                                        max_value = max(segment_values)
                                    else:
                                        avg_value = None
                                        max_value = None
                                    
                                    if current_affected:
                                        affected_length += segment_length_m
                                    else:
                                        unaffected_length += segment_length_m
                                    
                                    segment_row = row.to_dict()
                                    segment_row['geometry'] = segment_geom
                                    segment_row['length_m'] = segment_length_m
                                    segment_row['affected'] = current_affected
                                    segment_row['exposure_level_avg'] = avg_value
                                    segment_row['exposure_level_max'] = max_value
                                    segment_rows.append(segment_row)
                                
                                current_segment_points = [sampled_points[i-1], sampled_points[i]]
                                current_segment_indices = [i-1, i]
                                current_affected = point_affected[i]
                        
                        # Add final segment
                        if len(current_segment_points) >= 2:
                            segment_geom = LineString(current_segment_points)
                            # Calculate geodesic length in meters
                            segment_length_m = geod.line_length(*zip(*current_segment_points))
                            
                            # Calculate average and maximum exposure levels for segment
                            segment_values = [raster_values[j] for j in current_segment_indices if j < len(raster_values) and raster_values[j] is not None]
                            if len(segment_values) > 0:
                                avg_value = sum(segment_values) / len(segment_values)
                                max_value = max(segment_values)
                            else:
                                avg_value = None
                                max_value = None
                            
                            # Update length totals
                            if current_affected:
                                affected_length += segment_length_m
                            else:
                                unaffected_length += segment_length_m
                            
                            # Create segment row with all original attributes
                            segment_row = row.to_dict()
                            segment_row['geometry'] = segment_geom
                            segment_row['length_m'] = segment_length_m
                            segment_row['affected'] = current_affected
                            segment_row['exposure_level_avg'] = avg_value
                            segment_row['exposure_level_max'] = max_value
                            segment_rows.append(segment_row)
                            
                    except Exception as e:
                        # If sampling fails, log the error and create single unaffected segment for entire line
                        print(f"Warning: Raster sampling failed for line {idx}, part: {e}")
                        segment_row = row.to_dict()
                        segment_row['geometry'] = single_line
                        segment_row['length_m'] = total_length_m
                        segment_row['affected'] = False
                        segment_row['exposure_level_avg'] = None
                        segment_row['exposure_level_max'] = None
                        segment_rows.append(segment_row)
                        unaffected_length += total_length_m
            
            # Build new GeoDataFrame with segments
            if segment_rows:
                segment_gdf = gpd.GeoDataFrame(segment_rows, crs=infrastructure_gdf.crs)
                # Ensure 'affected' column is boolean type
                if 'affected' in segment_gdf.columns:
                    segment_gdf['affected'] = segment_gdf['affected'].astype(bool)
            else:
                # Empty result
                segment_gdf = gpd.GeoDataFrame(geometry=[], crs=infrastructure_gdf.crs)
            
            # Ensure meters are valid floats (not NaN)
            affected_meters = float(affected_length) if not np.isnan(affected_length) else 0.0
            unaffected_meters = float(unaffected_length) if not np.isnan(unaffected_length) else 0.0
            
            return {
                "affected_count": 0,
                "unaffected_count": 0,
                "affected_meters": affected_meters,
                "unaffected_meters": unaffected_meters,
                "full_gdf": segment_gdf  # Return segmented GDF
            }

