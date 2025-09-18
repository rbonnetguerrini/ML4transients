"""
Cross-matching module for ML4transients.

This module provides functionality to cross-match astronomical objects
with external catalogs (e.g., Gaia, PS1) based on sky coordinates.
Optimized for LSST diaObjectId-level matching.
"""

import numpy as np
import pandas as pd
import pickle
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial import cKDTree
import time
import os
from lsst.daf.butler import Butler
from lsst.geom import SpherePoint, degrees


class CrossMatcher:
    """
    Efficient cross-matching of astronomical objects with external catalogs.
    
    Supports multiple catalog formats and provides optimized spatial querying
    using KDTree for fast nearest-neighbor searches.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the CrossMatcher.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing catalog paths and parameters
        """
        self.config = config
        self.catalogs = {}
        self.trees = {}
        self.loaded_catalogs = set()
        
    def load_catalog(self, catalog_name: str, catalog_path: Union[str, Path], 
                    ra_col: str = 'ra', dec_col: str = 'dec'):
        """
        Load an external catalog for cross-matching.
        
        Parameters
        ----------
        catalog_name : str
            Name identifier for the catalog
        catalog_path : str or Path
            Path to catalog file (supports .pkl, .csv, .h5)
        ra_col : str
            Column name for right ascension (in degrees)
        dec_col : str  
            Column name for declination (in degrees)
        """
        print(f"Loading catalog '{catalog_name}' from {catalog_path}")
        catalog_path = Path(catalog_path)
        
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}")
        
        # Load catalog based on file extension
        if catalog_path.suffix == '.pkl':
            with open(catalog_path, 'rb') as f:
                catalog_df = pickle.load(f)
        elif catalog_path.suffix == '.csv':
            catalog_df = pd.read_csv(catalog_path)
        elif catalog_path.suffix == '.h5':
            catalog_df = pd.read_hdf(catalog_path)
        else:
            raise ValueError(f"Unsupported catalog format: {catalog_path.suffix}")
        
        # Validate coordinate columns
        if ra_col not in catalog_df.columns:
            raise ValueError(f"RA column '{ra_col}' not found in catalog")
        if dec_col not in catalog_df.columns:
            raise ValueError(f"Dec column '{dec_col}' not found in catalog")
        
        # Store catalog
        self.catalogs[catalog_name] = catalog_df
        
        # Create KDTree for fast spatial queries
        coords = np.vstack((catalog_df[ra_col], catalog_df[dec_col])).T
        self.trees[catalog_name] = cKDTree(coords)
        self.loaded_catalogs.add(catalog_name)
        
        print(f"  Loaded {len(catalog_df)} sources from {catalog_name}")
        
    def extract_object_coordinates(self, config: dict) -> pd.DataFrame:
        """
        Extract unique diaObjectId coordinates from LSST data.
        
        This function queries the LSST Butler to get coordinates for each
        unique diaObjectId, using the first observation of each object.
        
        Parameters
        ----------
        config : dict
            LSST data configuration (repo, collection, etc.)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns [diaObjectId, ra, dec, tract, patch]
        """
        print("Extracting diaObjectId coordinates from LSST data...")
        
        repo = config["repo"] 
        collection = config["collection"]
        butler = Butler(repo, collections=collection)
        registry = butler.registry
        
        # Query for all DIA source datasets
        print("Querying LSST datasets...")
        datasetRefs = list(registry.queryDatasets(
            datasetType='goodSeeingDiff_assocDiaSrcTable',
            collections=collection
        ))
        print(f"Found {len(datasetRefs)} datasets to process")
        
        object_coords = []
        total_refs = len(datasetRefs)
        start_time = time.time()
        
        for i, ref in enumerate(datasetRefs):
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_refs - i - 1) / rate if rate > 0 else 0
                print(f"  Processing {i+1}/{total_refs} - "
                      f"Rate: {rate:.1f}/s, ETA: {eta/60:.1f}min")
            
            try:
                # Load DIA source table
                dia_table = butler.get('goodSeeingDiff_assocDiaSrcTable', 
                                     dataId=ref.dataId)
                
                # Check if coordinate columns exist
                coord_columns = []
                if hasattr(dia_table, 'coord'):
                    # If coord is a SpherePoint object
                    for idx, row in enumerate(dia_table):
                        if hasattr(row, 'coord') and row.coord is not None:
                            coord = row.coord
                            ra = coord.getRa().asDegrees()
                            dec = coord.getDec().asDegrees()
                        else:
                            # Fallback: try to find ra/dec columns
                            ra = getattr(row, 'ra', None)
                            dec = getattr(row, 'dec', None)
                            if ra is None or dec is None:
                                continue
                        
                        object_coords.append({
                            'diaObjectId': getattr(row, 'diaObjectId', 
                                                 getattr(row, 'diaObjectId', None)),
                            'ra': ra,
                            'dec': dec,
                            'tract': ref.dataId['tract'],
                            'patch': ref.dataId['patch']
                        })
                else:
                    # Try direct column access
                    ra_cols = [col for col in dir(dia_table) if 'ra' in col.lower()]
                    dec_cols = [col for col in dir(dia_table) if 'dec' in col.lower()]
                    
                    if ra_cols and dec_cols:
                        ra_col = ra_cols[0]  # Take first RA column found
                        dec_col = dec_cols[0]  # Take first Dec column found
                        
                        for idx, row in enumerate(dia_table):
                            ra = getattr(row, ra_col, None)
                            dec = getattr(row, dec_col, None)
                            
                            if ra is not None and dec is not None:
                                object_coords.append({
                                    'diaObjectId': getattr(row, 'diaObjectId'),
                                    'ra': ra,
                                    'dec': dec,
                                    'tract': ref.dataId['tract'],
                                    'patch': ref.dataId['patch']
                                })
                
            except Exception as e:
                print(f"  Warning: Failed to process {ref.dataId}: {e}")
                continue
        
        if not object_coords:
            raise RuntimeError("No coordinate data found. Check LSST data schema.")
        
        # Create DataFrame and remove duplicates (keep first occurrence)
        print(f"Creating coordinate index with {len(object_coords)} entries...")
        coords_df = pd.DataFrame(object_coords)
        coords_df = coords_df.drop_duplicates(subset=['diaObjectId'], keep='first')
        coords_df['diaObjectId'] = coords_df['diaObjectId'].astype(np.int64)
        
        print(f"Extracted coordinates for {len(coords_df)} unique diaObjectIds")
        return coords_df
        
    def crossmatch_objects(self, object_coords: pd.DataFrame, 
                          catalog_name: str, tolerance_arcsec: float = 1.0) -> pd.DataFrame:
        """
        Cross-match diaObjectIds with an external catalog.
        
        Parameters
        ----------
        object_coords : pd.DataFrame
            DataFrame with columns [diaObjectId, ra/coord_ra, dec/coord_dec]
        catalog_name : str
            Name of loaded catalog to match against
        tolerance_arcsec : float
            Matching tolerance in arcseconds
            
        Returns
        -------
        pd.DataFrame
            Input DataFrame with added column 'in_{catalog_name}' (boolean)
        """
        if catalog_name not in self.loaded_catalogs:
            raise ValueError(f"Catalog '{catalog_name}' not loaded. "
                           f"Available: {list(self.loaded_catalogs)}")
        
        print(f"Cross-matching {len(object_coords)} objects with {catalog_name}")
        print(f"  Tolerance: {tolerance_arcsec} arcsec")
        
        # Convert tolerance to degrees
        tolerance_deg = tolerance_arcsec / 3600.0
        
        # Determine RA and Dec column names
        ra_col = 'coord_ra' if 'coord_ra' in object_coords.columns else 'ra'
        dec_col = 'coord_dec' if 'coord_dec' in object_coords.columns else 'dec'
        
        if ra_col not in object_coords.columns or dec_col not in object_coords.columns:
            raise ValueError(f"Required coordinate columns not found. Expected 'ra'/'coord_ra' and 'dec'/'coord_dec', "
                           f"got: {list(object_coords.columns)}")
        
        # Prepare coordinate arrays
        obj_coords = np.vstack((object_coords[ra_col], object_coords[dec_col])).T
        
        # Perform spatial query
        start_time = time.time()
        tree = self.trees[catalog_name]
        matches = tree.query_ball_point(obj_coords, r=tolerance_deg)
        
        # Create boolean array for matches
        has_match = np.array([len(match) > 0 for match in matches])
        
        # Add result to DataFrame
        result_df = object_coords.copy()
        result_df[f'in_{catalog_name}'] = has_match
        
        query_time = time.time() - start_time
        match_count = np.sum(has_match)
        match_rate = match_count / len(object_coords) * 100
        
        print(f"  Found {match_count}/{len(object_coords)} matches "
              f"({match_rate:.1f}%) in {query_time:.2f}s")
        
        return result_df
        
    def crossmatch_multiple_catalogs(self, object_coords: pd.DataFrame,
                                   catalog_specs: List[Dict]) -> pd.DataFrame:
        """
        Cross-match against multiple catalogs efficiently.
        
        Parameters
        ----------
        object_coords : pd.DataFrame
            DataFrame with object coordinates
        catalog_specs : List[Dict]
            List of catalog specifications, each containing:
            - 'name': catalog name
            - 'tolerance_arcsec': matching tolerance
            
        Returns
        -------
        pd.DataFrame
            DataFrame with boolean columns for each catalog
        """
        result_df = object_coords.copy()
        
        for spec in catalog_specs:
            catalog_name = spec['name']
            tolerance = spec.get('tolerance_arcsec', 1.0)
            
            result_df = self.crossmatch_objects(result_df, catalog_name, tolerance)
            
        return result_df
        
    def save_crossmatch_results(self, results_df: pd.DataFrame, 
                               output_path: Union[str, Path]):
        """
        Save cross-match results to file.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Cross-match results DataFrame
        output_path : str or Path
            Output file path (.h5, .csv, or .pkl)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.h5':
            results_df.to_hdf(output_path, key='crossmatch', mode='w', 
                            format='table', index=False)
        elif output_path.suffix == '.csv':
            results_df.to_csv(output_path, index=False)
        elif output_path.suffix == '.pkl':
            with open(output_path, 'wb') as f:
                pickle.dump(results_df, f)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
            
        print(f"Cross-match results saved to {output_path}")
        
    def load_crossmatch_results(self, results_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load previously saved cross-match results.
        
        Parameters
        ---------- 
        results_path : str or Path
            Path to results file
            
        Returns
        -------
        pd.DataFrame
            Cross-match results DataFrame
        """
        results_path = Path(results_path)
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
            
        if results_path.suffix == '.h5':
            return pd.read_hdf(results_path, key='crossmatch')
        elif results_path.suffix == '.csv':
            return pd.read_csv(results_path)
        elif results_path.suffix == '.pkl':
            with open(results_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {results_path.suffix}")


def perform_crossmatching(lsst_config: dict, catalog_paths: List[str], 
                         catalog_names: List[str] = None, 
                         tolerances_arcsec: List[float] = None,
                         ra_columns: List[str] = None,
                         dec_columns: List[str] = None):
    """
    Main function to perform cross-matching after lightcurve extraction.
    
    Parameters
    ----------
    lsst_config : dict
        LSST data configuration (repo, collection, path)
    catalog_paths : List[str]
        Paths to catalog files for cross-matching
    catalog_names : List[str], optional
        Names for each catalog. If None, uses filenames.
    tolerances_arcsec : List[float], optional
        Matching tolerances in arcseconds for each catalog. Default: 1.0 for all.
    ra_columns : List[str], optional
        RA column names for each catalog. Default: 'ra' for all.
    dec_columns : List[str], optional
        Dec column names for each catalog. Default: 'dec' for all.
        
    Returns
    -------
    pd.DataFrame
        Cross-match results
    """
    print("=== Starting Cross-Matching Process ===")
    
    if not catalog_paths:
        print("No catalogs specified for cross-matching")
        return None
    
    # Set defaults
    num_catalogs = len(catalog_paths)
    if catalog_names is None:
        catalog_names = [Path(p).stem for p in catalog_paths]
    if tolerances_arcsec is None:
        tolerances_arcsec = [1.0] * num_catalogs
    if ra_columns is None:
        ra_columns = ['ra'] * num_catalogs
    if dec_columns is None:
        dec_columns = ['dec'] * num_catalogs
    
    # Validate input lengths
    if not all(len(lst) == num_catalogs for lst in [catalog_names, tolerances_arcsec, ra_columns, dec_columns]):
        raise ValueError("All catalog parameter lists must have the same length as catalog_paths")
    
    # Initialize cross-matcher
    crossmatcher = CrossMatcher({})
    
    # Load external catalogs
    for i, (path, name, ra_col, dec_col) in enumerate(zip(catalog_paths, catalog_names, ra_columns, dec_columns)):
        print(f"Loading catalog {i+1}/{num_catalogs}: {name}")
        crossmatcher.load_catalog(name, path, ra_col, dec_col)
    
    # Extract object coordinates from LSST data
    object_coords = crossmatcher.extract_object_coordinates(lsst_config)
    
    # Perform cross-matching
    catalog_specs = [
        {
            'name': name,
            'tolerance_arcsec': tolerance
        }
        for name, tolerance in zip(catalog_names, tolerances_arcsec)
    ]
    
    results = crossmatcher.crossmatch_multiple_catalogs(object_coords, catalog_specs)
    
    # Save results
    output_path = Path(lsst_config['path']) / 'crossmatch' / 'crossmatch_results.h5'
    crossmatcher.save_crossmatch_results(results, output_path)
    
    # Create summary
    summary = {
        'total_objects': len(results),
        'catalogs_matched': len(catalog_names),
        'processing_time': time.time(),
        'catalog_info': {
            'names': catalog_names,
            'paths': catalog_paths,
            'tolerances_arcsec': tolerances_arcsec
        }
    }
    
    for name in catalog_names:
        col_name = f'in_{name}'
        if col_name in results.columns:
            match_count = results[col_name].sum()
            match_rate = match_count / len(results) * 100
            summary[f'{name}_matches'] = match_count
            summary[f'{name}_match_rate_percent'] = match_rate
            print(f"Summary - {name}: {match_count}/{len(results)} "
                  f"objects matched ({match_rate:.1f}%)")
    
    # Save summary
    summary_path = output_path.parent / 'crossmatch_summary.pkl'
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"Cross-matching completed. Results saved to {output_path}")
    return results