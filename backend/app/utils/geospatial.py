"""
Geospatial utility functions
"""

import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
from shapely.geometry import Point, LineString
from typing import Optional, Callable
from pyproj import Geod
import csv

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


def parse_vulnerability_curve(csv_path: str) -> Callable[[float], float]:
    """
    Parse a vulnerability curve CSV file and return an interpolation function.
    
    CSV format: Two columns (intensity, proportion_destroyed)
    - First column: hazard intensity
    - Second column: proportion destroyed (0-1)
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Interpolation function that takes intensity and returns proportion destroyed
    """
    # Try multiple encodings
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
    delimiter = ','
    encoding = 'utf-8'
    
    for enc in encodings:
        try:
            with open(csv_path, 'r', encoding=enc) as f:
                first_line = f.readline()
                sniffer = csv.Sniffer()
                try:
                    delimiter = sniffer.sniff(first_line, delimiters=',;').delimiter
                except:
                    delimiter = ';' if ';' in first_line else ','
                encoding = enc
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # Read CSV
    df = pd.read_csv(csv_path, sep=delimiter, encoding=encoding, header=None)
    
    if df.shape[1] < 2:
        raise ValueError("Vulnerability curve CSV must have at least 2 columns")
    
    # Extract intensity and proportion_destroyed (first two columns)
    intensity = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    proportion_destroyed = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    
    # Remove rows with NaN
    valid_mask = ~(intensity.isna() | proportion_destroyed.isna())
    intensity = intensity[valid_mask].values
    proportion_destroyed = proportion_destroyed[valid_mask].values
    
    if len(intensity) == 0:
        raise ValueError("No valid data rows found in vulnerability curve CSV")
    
    # Validate proportion_destroyed is in [0, 1]
    if (proportion_destroyed < 0).any() or (proportion_destroyed > 1).any():
        raise ValueError("Proportion destroyed values must be between 0 and 1")
    
    # Sort by intensity
    sort_idx = np.argsort(intensity)
    intensity = intensity[sort_idx]
    proportion_destroyed = proportion_destroyed[sort_idx]
    
    # Create interpolation function
    # For values below minimum, return 0 (or first value)
    # For values above maximum, return 1 (or last value)
    # For values in range, use linear interpolation
    def interpolate(intensity_value: float) -> float:
        if np.isnan(intensity_value):
            return 0.0
        
        if intensity_value <= intensity[0]:
            return float(proportion_destroyed[0])
        if intensity_value >= intensity[-1]:
            return float(proportion_destroyed[-1])
        
        # Linear interpolation
        return float(np.interp(intensity_value, intensity, proportion_destroyed))
    
    return interpolate


def analyze_intersection(
    infrastructure_gdf: gpd.GeoDataFrame,
    hazard_raster_path: str,
    geometry_type: str,
    intensity_threshold: Optional[float] = None,
    cached_raster_values: Optional[np.ndarray] = None,
    vulnerability_curve_interp: Optional[Callable[[float], float]] = None,
    replacement_value: Optional[float] = None
) -> dict:
    """
    Analyze intersection between infrastructure and hazard raster
    
    All hazard rasters use EPSG:4326 (WGS84) as their CRS.
    
    Args:
        infrastructure_gdf: GeoDataFrame with infrastructure assets
        hazard_raster_path: Path or URL to hazard raster (assumed EPSG:4326)
        geometry_type: Geometry type ("Point" or "LineString") from stored metadata
        intensity_threshold: Optional threshold for filtering hazard intensity
        cached_raster_values: Optional pre-sampled raster values (for threshold changes)
        
    Returns:
        Dictionary with analysis results:
        - affected_count/unaffected_count (for points)
        - affected_meters/unaffected_meters (for lines)
        - full_gdf: GeoDataFrame with all features and affected status
        - raster_values: (for points) sampled values for caching
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
            from rasterio.windows import from_bounds
            from rasterio.transform import rowcol
            
            n_points = len(infrastructure_gdf)
            
            # Check if we have cached raster values (for threshold changes)
            if cached_raster_values is not None and len(cached_raster_values) == n_points:
                raster_values = cached_raster_values
            else:
                # Extract coordinates as numpy arrays
                x_coords = infrastructure_gdf.geometry.x.to_numpy()
                y_coords = infrastructure_gdf.geometry.y.to_numpy()
                
                # Use tile-based sampling: group points by tiles, read each tile once
                # This minimizes HTTP requests for remote COGs with scattered points
                tile_size = 1.0  # 1 degree tiles (~120x120 pixels for 30 arc-second data)
                
                # Assign each point to a tile
                tile_x = np.floor(x_coords / tile_size).astype(np.int32)
                tile_y = np.floor(y_coords / tile_size).astype(np.int32)
                
                # Create unique tile keys (offset to handle negative coords)
                tile_keys = (tile_x + 180).astype(np.int64) * 1000 + (tile_y + 90).astype(np.int64)
                unique_tiles = np.unique(tile_keys)
                n_tiles = len(unique_tiles)
                
                # Sample by reading tiles in parallel (much faster for remote COGs)
                raster_values = np.full(n_points, np.nan, dtype=np.float64)
                raster_url = hazard_raster_path  # Capture for use in threads
                
                def process_tile(tile_key):
                    """Process a single tile - reads tile and samples points."""
                    # Find points in this tile
                    mask = tile_keys == tile_key
                    point_indices = np.where(mask)[0]
                    
                    # Reconstruct tile bounds
                    tx = (tile_key // 1000) - 180
                    ty = (tile_key % 1000) - 90
                    tile_minx = tx * tile_size
                    tile_maxx = tile_minx + tile_size
                    tile_miny = ty * tile_size
                    tile_maxy = tile_miny + tile_size
                    
                    try:
                        # Each thread opens its own connection (thread-safe)
                        with rasterio.open(raster_url) as tile_src:
                            window = from_bounds(tile_minx, tile_miny, tile_maxx, tile_maxy, tile_src.transform)
                            data = tile_src.read(1, window=window, boundless=True, fill_value=np.nan)
                            
                            if data.size == 0:
                                return point_indices, np.full(len(point_indices), np.nan)
                            
                            window_transform = tile_src.window_transform(window)
                            
                            # Get coordinates of points in this tile
                            tile_x_coords = x_coords[point_indices]
                            tile_y_coords = y_coords[point_indices]
                            
                            # Convert to pixel indices
                            rows, cols = rowcol(window_transform, tile_x_coords, tile_y_coords)
                            rows = np.clip(np.array(rows), 0, data.shape[0] - 1).astype(int)
                            cols = np.clip(np.array(cols), 0, data.shape[1] - 1).astype(int)
                            
                            return point_indices, data[rows, cols]
                            
                    except Exception as e:
                        return point_indices, np.full(len(point_indices), np.nan)
                
                # Process tiles in parallel
                from concurrent.futures import ThreadPoolExecutor, as_completed
                max_workers = min(16, n_tiles)  # Cap at 16 concurrent connections
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_tile, tk): tk for tk in unique_tiles}
                    for future in as_completed(futures):
                        point_indices, values = future.result()
                        raster_values[point_indices] = values
            
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
            
            # Calculate vulnerability and damage cost if vulnerability analysis is enabled
            total_damage_cost = 0.0
            total_replacement_value = 0.0
            if vulnerability_curve_interp is not None and replacement_value is not None:
                vulnerability_values = []
                damage_cost_values = []
                
                # For points, total replacement value = replacement_value * number of points
                total_replacement_value = replacement_value * len(infrastructure_gdf)
                
                for idx, row in infrastructure_gdf.iterrows():
                    exposure = row.get('exposure_level')
                    if exposure is not None and not np.isnan(exposure):
                        vulnerability = vulnerability_curve_interp(float(exposure))
                        damage_cost = replacement_value * vulnerability
                    else:
                        vulnerability = 0.0
                        damage_cost = 0.0
                    
                    vulnerability_values.append(vulnerability)
                    damage_cost_values.append(damage_cost)
                    total_damage_cost += damage_cost
                
                infrastructure_gdf['vulnerability'] = vulnerability_values
                infrastructure_gdf['damage_cost'] = damage_cost_values
            else:
                infrastructure_gdf['vulnerability'] = None
                infrastructure_gdf['damage_cost'] = None
            
            affected_count = int(affected_mask.sum())
            unaffected_count = len(infrastructure_gdf) - affected_count
            
            result = {
                "affected_count": affected_count,
                "unaffected_count": unaffected_count,
                "affected_meters": 0.0,
                "unaffected_meters": 0.0,
                "full_gdf": infrastructure_gdf,  # Return full GDF with affected status
                "raster_values": raster_values  # Return for caching (threshold changes)
            }
            
            if vulnerability_curve_interp is not None and replacement_value is not None:
                result["total_damage_cost"] = total_damage_cost
                result["total_replacement_value"] = total_replacement_value
            
            return result
        
        else:  # LineString
            from rasterio.windows import from_bounds
            from rasterio.transform import rowcol
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            geod = Geod(ellps="WGS84")
            
            # Check if we have cached data (for threshold changes)
            use_cache = (cached_raster_values is not None and 
                        isinstance(cached_raster_values, dict) and 
                        'line_data' in cached_raster_values and 
                        'raster_values' in cached_raster_values)
            
            if use_cache:
                line_data = cached_raster_values['line_data']
                all_raster_values = cached_raster_values['raster_values']
            else:
                # Phase 1: Collect all sample points from all lines
                line_data = []
                all_sample_points = []
                
                for idx, row in infrastructure_gdf.iterrows():
                    line = row.geometry
                    row_dict = row.to_dict()
                    
                    # Handle MultiLineString
                    if line.geom_type == 'MultiLineString':
                        lines = list(line.geoms)
                    else:
                        lines = [line]
                    
                    for single_line in lines:
                        line_coords = list(single_line.coords)
                        
                        if len(line_coords) < 2:
                            continue
                        
                        total_length_m = geod.line_length(*zip(*line_coords))
                        if total_length_m == 0:
                            continue
                        
                        # Sample points along line at 100m intervals
                        interval_meters = 100.0
                        sampled_points = [line_coords[0]]
                        
                        current_distance = 0.0
                        while current_distance < total_length_m:
                            current_distance += interval_meters
                            if current_distance >= total_length_m:
                                sampled_points.append(line_coords[-1])
                                break
                            normalized_dist = current_distance / total_length_m
                            point = single_line.interpolate(normalized_dist, normalized=True)
                            sampled_points.append((point.x, point.y))
                        
                        if sampled_points[-1] != line_coords[-1]:
                            sampled_points.append(line_coords[-1])
                        
                        if len(sampled_points) < 2:
                            continue
                        
                        # Store line data
                        line_data.append({
                            'row_dict': row_dict,
                            'sampled_points': sampled_points,
                            'total_length_m': total_length_m,
                            'single_line': single_line
                        })
                        
                        # Add points to batch
                        for pt in sampled_points:
                            all_sample_points.append(pt)
                
                n_points = len(all_sample_points)
                
                if n_points == 0:
                    return {
                        "affected_count": 0,
                        "unaffected_count": 0,
                        "affected_meters": 0.0,
                        "unaffected_meters": 0.0,
                        "full_gdf": gpd.GeoDataFrame(geometry=[], crs=infrastructure_gdf.crs)
                    }
                
                # Phase 2: Tile-based parallel sampling
                x_coords = np.array([p[0] for p in all_sample_points])
                y_coords = np.array([p[1] for p in all_sample_points])
                
                tile_size = 1.0
                tile_x = np.floor(x_coords / tile_size).astype(np.int32)
                tile_y = np.floor(y_coords / tile_size).astype(np.int32)
                tile_keys = (tile_x + 180).astype(np.int64) * 1000 + (tile_y + 90).astype(np.int64)
                unique_tiles = np.unique(tile_keys)
                n_tiles = len(unique_tiles)
                
                all_raster_values = np.full(n_points, np.nan, dtype=np.float64)
                raster_url = hazard_raster_path
                
                def process_tile(tile_key):
                    mask = tile_keys == tile_key
                    point_indices = np.where(mask)[0]
                    
                    tx = (tile_key // 1000) - 180
                    ty = (tile_key % 1000) - 90
                    tile_minx = tx * tile_size
                    tile_maxx = tile_minx + tile_size
                    tile_miny = ty * tile_size
                    tile_maxy = tile_miny + tile_size
                    
                    try:
                        with rasterio.open(raster_url) as tile_src:
                            window = from_bounds(tile_minx, tile_miny, tile_maxx, tile_maxy, tile_src.transform)
                            data = tile_src.read(1, window=window, boundless=True, fill_value=np.nan)
                            
                            if data.size == 0:
                                return point_indices, np.full(len(point_indices), np.nan)
                            
                            window_transform = tile_src.window_transform(window)
                            tile_x_coords = x_coords[point_indices]
                            tile_y_coords = y_coords[point_indices]
                            
                            rows, cols = rowcol(window_transform, tile_x_coords, tile_y_coords)
                            rows = np.clip(np.array(rows), 0, data.shape[0] - 1).astype(int)
                            cols = np.clip(np.array(cols), 0, data.shape[1] - 1).astype(int)
                            
                            return point_indices, data[rows, cols]
                    except Exception:
                        return point_indices, np.full(len(point_indices), np.nan)
                
                max_workers = min(16, n_tiles)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_tile, tk): tk for tk in unique_tiles}
                    for future in as_completed(futures):
                        point_indices, values = future.result()
                        all_raster_values[point_indices] = values
                
                # Build per-line raster values from the batch results
                current_point_idx = 0
                for ld in line_data:
                    n_pts = len(ld['sampled_points'])
                    ld['raster_values'] = all_raster_values[current_point_idx:current_point_idx + n_pts]
                    current_point_idx += n_pts
            
            # Phase 3: Process each line with pre-sampled values (always runs)
            affected_length = 0.0
            unaffected_length = 0.0
            total_damage_cost = 0.0
            total_replacement_value = 0.0
            segment_rows = []
            
            # Calculate total original length of all lines for replacement value calculation
            # This must be done before processing segments to get the full original length
            total_original_length = 0.0
            if vulnerability_curve_interp is not None and replacement_value is not None:
                for ld in line_data:
                    total_original_length += ld['total_length_m']
                total_replacement_value = replacement_value * total_original_length
            
            for ld in line_data:
                row_dict = ld['row_dict']
                sampled_points = ld['sampled_points']
                raster_values = ld['raster_values']
                
                # Classify points as affected
                if intensity_threshold is not None:
                    point_affected = [not np.isnan(v) and v >= intensity_threshold for v in raster_values]
                else:
                    point_affected = [not np.isnan(v) and v > 0 for v in raster_values]
                
                # Create segments by grouping consecutive points with same status
                current_segment_points = [sampled_points[0]]
                current_segment_indices = [0]
                current_affected = point_affected[0]
                
                def create_segment(points, indices, affected):
                    nonlocal affected_length, unaffected_length, total_damage_cost
                    if len(points) < 2:
                        return
                    
                    segment_geom = LineString(points)
                    segment_length_m = geod.line_length(*zip(*points))
                    
                    segment_vals = [raster_values[j] for j in indices if not np.isnan(raster_values[j])]
                    avg_value = sum(segment_vals) / len(segment_vals) if segment_vals else None
                    max_value = max(segment_vals) if segment_vals else None
                    
                    # Calculate distance-weighted average vulnerability for this segment
                    vulnerability = None
                    damage_cost = None
                    if vulnerability_curve_interp is not None and replacement_value is not None:
                        if segment_vals:
                            # Calculate distance-weighted average vulnerability
                            # For each segment between consecutive points, calculate vulnerability
                            # at the segment midpoint (average of endpoints) and weight by segment length
                            vulnerability_sum = 0.0
                            length_sum = 0.0
                            
                            for i in range(len(points) - 1):
                                idx1 = indices[i]
                                idx2 = indices[i + 1]
                                
                                # Get exposure values at both endpoints
                                val1 = raster_values[idx1] if idx1 < len(raster_values) and not np.isnan(raster_values[idx1]) else None
                                val2 = raster_values[idx2] if idx2 < len(raster_values) and not np.isnan(raster_values[idx2]) else None
                                
                                # Calculate segment length
                                pt1 = points[i]
                                pt2 = points[i + 1]
                                # geod.line_length expects two arrays: lons and lats
                                # Use the same pattern as elsewhere: *zip(*coords)
                                seg_len = geod.line_length(*zip(*[pt1, pt2]))
                                
                                # Calculate vulnerability for this segment
                                if val1 is not None and val2 is not None:
                                    # Average vulnerability of both endpoints
                                    vuln1 = vulnerability_curve_interp(float(val1))
                                    vuln2 = vulnerability_curve_interp(float(val2))
                                    avg_vuln = (vuln1 + vuln2) / 2.0
                                elif val1 is not None:
                                    avg_vuln = vulnerability_curve_interp(float(val1))
                                elif val2 is not None:
                                    avg_vuln = vulnerability_curve_interp(float(val2))
                                else:
                                    avg_vuln = 0.0
                                
                                vulnerability_sum += avg_vuln * seg_len
                                length_sum += seg_len
                            
                            if length_sum > 0:
                                vulnerability = vulnerability_sum / length_sum
                            else:
                                vulnerability = 0.0
                            
                            # Damage cost = replacement_value (per meter) × vulnerability × length
                            damage_cost = replacement_value * vulnerability * segment_length_m
                            total_damage_cost += damage_cost
                        else:
                            vulnerability = 0.0
                            damage_cost = 0.0
                    
                    if affected:
                        affected_length += segment_length_m
                    else:
                        unaffected_length += segment_length_m
                    
                    seg_row = row_dict.copy()
                    seg_row['geometry'] = segment_geom
                    seg_row['length_m'] = segment_length_m
                    seg_row['affected'] = affected
                    seg_row['exposure_level_avg'] = avg_value
                    seg_row['exposure_level_max'] = max_value
                    if vulnerability is not None:
                        seg_row['vulnerability'] = vulnerability
                    if damage_cost is not None:
                        seg_row['damage_cost'] = damage_cost
                    segment_rows.append(seg_row)
                
                for i in range(1, len(sampled_points)):
                    if point_affected[i] == current_affected:
                        current_segment_points.append(sampled_points[i])
                        current_segment_indices.append(i)
                    else:
                        create_segment(current_segment_points, current_segment_indices, current_affected)
                        current_segment_points = [sampled_points[i-1], sampled_points[i]]
                        current_segment_indices = [i-1, i]
                        current_affected = point_affected[i]
                
                # Final segment
                create_segment(current_segment_points, current_segment_indices, current_affected)
            
            # Build GeoDataFrame
            if segment_rows:
                segment_gdf = gpd.GeoDataFrame(segment_rows, crs=infrastructure_gdf.crs)
                if 'affected' in segment_gdf.columns:
                    segment_gdf['affected'] = segment_gdf['affected'].astype(bool)
            else:
                segment_gdf = gpd.GeoDataFrame(geometry=[], crs=infrastructure_gdf.crs)
            
            affected_meters = float(affected_length) if not np.isnan(affected_length) else 0.0
            unaffected_meters = float(unaffected_length) if not np.isnan(unaffected_length) else 0.0
            
            # total_replacement_value is already calculated above using total_original_length
            
            result = {
                "affected_count": 0,
                "unaffected_count": 0,
                "affected_meters": affected_meters,
                "unaffected_meters": unaffected_meters,
                "full_gdf": segment_gdf,
                "line_data": line_data,
                "raster_values": all_raster_values
            }
            
            if vulnerability_curve_interp is not None and replacement_value is not None:
                result["total_damage_cost"] = total_damage_cost
                result["total_replacement_value"] = total_replacement_value
            
            return result

