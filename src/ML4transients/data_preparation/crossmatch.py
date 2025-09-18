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

