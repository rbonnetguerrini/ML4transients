"""
Fake Source Injection Module for LSST Transient Detection

This module provides functionality for injecting artificial transient sources into LSST images for machine learning training and testing. It supports both galaxy-hosted and hostless transients with realistic magnitude distributions and geometric placement.

The module handles:
- Magnitude distribution sampling from observed data
- Geometric calculations for galaxy shapes and orientations  
- Coordinate transformations between pixel and sky coordinates
- Batch processing of multiple CCDs and visits
- Catalog creation and management

Classes:
    InjectionParams: Data class for injection parameters
    Position: Data class for position information
    InjectionConfig: Configuration for injection generation
    MagnitudeDistribution: Handles magnitude sampling from distributions
    GeometryCalculator: Geometric calculations for galaxies and injections
    InjectionGenerator: Main class for generating fake source injections
    CatalogProcessor: Handles catalog creation and processing

Example:
    >>> config = InjectionConfig(galaxy_fraction=0.8, min_injections=100)
    >>> processor = CatalogProcessor()
    >>> ra, dec, mags = processor.process_catalog(galaxy_catalog, image, 'r')
"""

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
import lsst.afw.display as afwDisplay
import lsst.geom
from astropy.table import Table
from lsst.geom import Point2D, Box2I, Point2I, Extent2I
import random
from scipy.stats import norm 
from lsst.source.injection import VisitInjectConfig, VisitInjectTask
from matplotlib.patches import Ellipse
import pickle
import scipy.stats as stats
import warnings
from typing import Tuple, List, Optional, Dict, Any
from functools import lru_cache
from dataclasses import dataclass

# Suppress common warnings that don't affect functionality
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*numpy._core.*")

# Constants
PIXEL_SCALE = 0.168432  # arcsec/pixel - LSST pixel scale
DEFAULT_FLUX_TYPE = 'calibFlux'  # Default flux column name in catalogs
MAG_DISTRIB = '/sps/lsst/users/rbonnetguerrini/ML4transients/saved/mag_distrib.json'  # Path to magnitude distributions

@dataclass
class InjectionParams:
    """
    Data class to hold injection parameters for a single fake source.
    
    Attributes:
        r (float): Radial distance from host center in pixels
        theta (float): Angular position in degrees (0-180)
        mag (float): Magnitude of the injected source in AB system
        orientation (float): Orientation angle of the host galaxy in radians
    """
    r: float
    theta: float
    mag: float
    orientation: float

@dataclass
class Position:
    """
    Data class to hold position information in different coordinate systems.
    
    Attributes:
        ra (float): Right ascension in degrees
        dec (float): Declination in degrees  
        x (Optional[float]): Pixel x-coordinate (optional)
        y (Optional[float]): Pixel y-coordinate (optional)
    """
    ra: float
    dec: float
    x: Optional[float] = None
    y: Optional[float] = None

@dataclass
class InjectionConfig:
    """
    Configuration parameters for injection generation.
    
    Attributes:
        galaxy_fraction (float): Fraction of injections to place near galaxies (0.0-1.0)
        hostless_fraction (float): Fraction of additional hostless injections
        min_injections (int): Minimum number of injections per CCD
        max_injections (Optional[int]): Maximum number of injections per CCD
        random_seed (int): Random seed for reproducible results
    """
    galaxy_fraction: float = 0.8
    hostless_fraction: float = 0.05
    min_injections: int = 50
    max_injections: Optional[int] = None
    random_seed: int = 42

class MagnitudeDistribution:
    """
    Handles magnitude distribution sampling for different photometric bands.
    
    This class loads pre-computed magnitude distributions from observed data
    and provides methods to sample realistic magnitudes for injected sources.
    
    Attributes:
        distribution_file (str): Path to JSON file containing distributions
        distributions (dict): Loaded distribution parameters by band
    
    Example:
        >>> mag_dist = MagnitudeDistribution()
        >>> magnitudes = mag_dist.sample('r', num_samples=100)
    """
    
    def __init__(self, distribution_file: str = MAG_DISTRIB):
        """
        Initialize magnitude distribution handler.
        
        Args:
            distribution_file (str): Path to JSON file with band distributions
        """
        self.distribution_file = distribution_file
        self.distributions = self._load_distributions()
    
    def _load_distributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Load magnitude distributions from JSON file.
        
        Returns:
            Dict containing distribution parameters for each band.
            Format: {band: {'distribution': dist_name, 'params': [param1, param2, ...]}}
            
        Note:
            Falls back to default normal distributions if file loading fails.
        """
        try:
            import json
            with open(self.distribution_file, "r") as f:
                data = json.load(f)
                print(f"Successfully loaded bands: {list(data.keys())}")
                return data
        except Exception as e:
            print(f"Could not load {self.distribution_file}: {e}")
            # Fallback to reasonable default distributions for LSST bands
            return {
                'g': {'distribution': 'norm', 'params': [22.0, 1.0]},
                'r': {'distribution': 'norm', 'params': [21.5, 1.0]},
                'i': {'distribution': 'norm', 'params': [21.0, 1.0]},
                'z': {'distribution': 'norm', 'params': [20.5, 1.0]},
                'y': {'distribution': 'norm', 'params': [20.0, 1.0]},
                'N921': {'distribution': 'norm', 'params': [21.0, 1.0]}
            }
    
    def sample(self, band: str, num_samples: int = 1) -> np.ndarray:
        """
        Sample magnitudes from the distribution for a given band.
        
        Args:
            band (str): Photometric band ('g', 'r', 'i', 'z', 'y')
            num_samples (int): Number of magnitude samples to generate
            
        Returns:
            np.ndarray: Array of sampled magnitudes in AB system
            
        Note:
            Returns default magnitude 22.0 if band not found or sampling fails.
        """
        if band not in self.distributions:
            print(f"Band '{band}' not found, using default magnitude 22.0")
            return np.full(num_samples, 22.0)
        
        try:
            best_distribution = self.distributions[band]
            dist_name = best_distribution['distribution']
            params = best_distribution['params']
            
            # Get the scipy distribution and sample from it
            dist = getattr(stats, dist_name)
            samples = dist.rvs(*params, size=num_samples)
            return np.asarray(samples, dtype=float)
        except Exception as e:
            print(f"Sampling failed for band {band}: {e}")
            return np.full(num_samples, 22.0)

class GeometryCalculator:
    """
    Handles geometric calculations for galaxy shapes and injection positioning.
    
    This class provides static methods for calculating galaxy orientations,
    coordinate transformations, and geometric operations needed for realistic
    source injection placement.
    """
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def calculate_orientation_and_extent(ixx: float, ixy: float, iyy: float) -> Tuple[float, float, float]:
        """
        Calculate galaxy orientation and extent from second moments.
        
        This method computes the orientation angle and semi-major/minor axes
        from the second moment matrix elements (ixx, ixy, iyy).
        
        Args:
            ixx (float): Second moment xx component (arcsec²)
            ixy (float): Second moment xy component (arcsec²)  
            iyy (float): Second moment yy component (arcsec²)
            
        Returns:
            Tuple[float, float, float]: (orientation, semi_major_axis, semi_minor_axis)
                - orientation: Angle in radians from x-axis
                - semi_major_axis: Length of semi-major axis (arcsec)
                - semi_minor_axis: Length of semi-minor axis (arcsec)
                
        Note:
            Uses eigenvalue decomposition of the moment tensor.
            Results are cached for performance with repeated calculations.
        """
        # Calculate orientation angle from off-diagonal moment
        theta = 0.5 * np.arctan2(2 * ixy, ixx - iyy)
        
        # Calculate eigenvalues of the moment tensor
        term1 = (ixx + iyy) / 2  # Average of diagonal elements
        term2 = np.sqrt(((ixx - iyy) / 2) ** 2 + ixy ** 2)  # Discriminant term
        lambda1 = term1 + term2  # Larger eigenvalue
        lambda2 = term1 - term2  # Smaller eigenvalue
        
        # Convert eigenvalues to semi-axes lengths
        a = np.sqrt(lambda1)  # Semi-major axis
        b = np.sqrt(lambda2)  # Semi-minor axis
        
        return theta, a, b
    
    @staticmethod
    def polar_to_cartesian(r: float, theta: float) -> Tuple[float, float]:
        """
        Convert polar coordinates to cartesian coordinates.
        
        Args:
            r (float): Radial distance
            theta (float): Angle in degrees
            
        Returns:
            Tuple[float, float]: (x, y) cartesian coordinates
        """
        x = r * np.cos(np.radians(theta))
        y = r * np.sin(np.radians(theta))
        return x, y
    
    @staticmethod
    def rotate_coordinates(x: float, y: float, orientation: float) -> Tuple[float, float]:
        """
        Rotate coordinates by a given orientation angle.
        
        Args:
            x (float): X-coordinate to rotate
            y (float): Y-coordinate to rotate
            orientation (float): Rotation angle in radians
            
        Returns:
            Tuple[float, float]: Rotated (x, y) coordinates
        """
        cos_o, sin_o = np.cos(orientation), np.sin(orientation)
        x_rot = x * cos_o - y * sin_o
        y_rot = x * sin_o + y * cos_o
        return x_rot, y_rot

class InjectionGenerator:
    """
    Main class for generating fake source injection parameters.
    
    This class handles the generation of positions and magnitudes for fake
    sources, supporting both galaxy-hosted and hostless transients with
    realistic distributions.
    
    Attributes:
        mag_dist (MagnitudeDistribution): Magnitude distribution handler
        geom_calc (GeometryCalculator): Geometry calculation utilities
    """
    
    def __init__(self, mag_dist_file: str = MAG_DISTRIB):
        """
        Initialize injection generator.
        
        Args:
            mag_dist_file (str): Path to magnitude distribution file
        """
        self.mag_dist = MagnitudeDistribution(mag_dist_file)
        self.geom_calc = GeometryCalculator()
        
    def generate_injection_params(self, galaxy_data: Dict[str, Any], band: str) -> InjectionParams:
        """
        Generate injection parameters for a single galaxy-hosted source.
        
        Args:
            galaxy_data (Dict[str, Any]): Galaxy properties including moments and flux
            band (str): Photometric band for magnitude sampling
            
        Returns:
            InjectionParams: Complete injection parameters for one source
            
        Note:
            Places sources with Gaussian distribution around galaxy center,
            with width proportional to galaxy size.
        """
        # Calculate galaxy geometry from second moments
        orientation, a, b = self.geom_calc.calculate_orientation_and_extent(
            galaxy_data['ixx'], galaxy_data['ixy'], galaxy_data['iyy']
        )
        
        # Sample position relative to galaxy center
        # Use normal distribution with width = semi-major axis
        r = np.random.normal(0, a)
        theta = random.uniform(0, 180)  # Random angle
        
        # Calculate magnitude from galaxy flux or sample from distribution
        flux = galaxy_data[DEFAULT_FLUX_TYPE]
        mag = (flux * u.nJy).to(u.ABmag)
        
        if np.isfinite(mag.value):
            # Vary magnitude relative to host galaxy brightness
            mag_value = random.uniform(mag.value - 1, mag.value + 3)
        else:
            # Fall back to distribution sampling
            mag_value = self.mag_dist.sample(band)[0]
        
        return InjectionParams(r, theta, mag_value, orientation)
    
    def generate_positions_vectorized(self, catalog: pd.DataFrame, band: str) -> Tuple[List[float], List[float], List[float]]:
        """
        Generate injection positions and magnitudes for multiple galaxies efficiently.
        
        This method uses vectorized operations to process entire catalogs at once,
        significantly improving performance over individual galaxy processing.
        
        Args:
            catalog (pd.DataFrame): Galaxy catalog with required columns:
                - ixx, ixy, iyy: Second moment components
                - coord_ra/ra, coord_dec/dec: Coordinates  
                - calibFlux: Calibrated flux measurements
            band (str): Photometric band for magnitude sampling
            
        Returns:
            Tuple[List[float], List[float], List[float]]: 
                (inject_ra, inject_dec, magnitudes) for all injections
                
        Note:
            Handles coordinate column name variations and provides fallbacks
            for failed calculations. Returns empty lists on major errors.
        """
        n_galaxies = len(catalog)
        
        try:
            # Pre-calculate all galaxy orientations and extents
            orientations = []
            a_values = []
            
            for i, (_, row) in enumerate(catalog.iterrows()):
                try:
                    orientation, a, b = self.geom_calc.calculate_orientation_and_extent(
                        row['ixx'], row['ixy'], row['iyy']
                    )
                    orientations.append(orientation)
                    a_values.append(a)
                except Exception as e:
                    print(f"Warning: geometry calculation failed for row {i}: {e}")
                    orientations.append(0.0)  # default orientation
                    a_values.append(1.0)      # default size
            
            # Convert to numpy arrays for vectorized operations
            orientations = np.array(orientations)
            a_values = np.array(a_values)
            
            # Vectorized position sampling
            try:
                # Sample radial distances proportional to galaxy sizes
                r_values = np.random.normal(0, a_values)
                # Random angles for all galaxies
                theta_values = np.random.uniform(0, 180, n_galaxies)
            except Exception as e:
                print(f"Error in random sampling: {e}")
                # Fallback to basic random sampling
                r_values = np.random.normal(0, 1, n_galaxies)
                theta_values = np.random.uniform(0, 180, n_galaxies)
            
            # Convert polar to cartesian coordinates (vectorized)
            x_values = r_values * np.cos(np.radians(theta_values))
            y_values = r_values * np.sin(np.radians(theta_values))
            
            # Apply galaxy orientations (vectorized rotation)
            cos_o = np.cos(orientations)
            sin_o = np.sin(orientations)
            x_rot = x_values * cos_o - y_values * sin_o
            y_rot = x_values * sin_o + y_values * cos_o
            
            # Handle different coordinate column naming conventions
            ra_values = catalog['coord_ra'].values if 'coord_ra' in catalog.columns else catalog['ra'].values
            dec_values = catalog['coord_dec'].values if 'coord_dec' in catalog.columns else catalog['dec'].values
            
            # Convert pixel offsets to sky coordinate offsets
            # Account for declination-dependent RA scaling
            delta_ra = (x_rot / np.cos(np.radians(dec_values))) * PIXEL_SCALE / 3600
            delta_dec = y_rot * PIXEL_SCALE / 3600
            
            # Apply offsets to host coordinates
            inject_ra = (ra_values - delta_ra).tolist()
            inject_dec = (dec_values - delta_dec).tolist()
            
            # Generate magnitudes for each injection
            mags = []
            for _, row in catalog.iterrows():
                try:
                    flux = row[DEFAULT_FLUX_TYPE]
                    mag = (flux * u.nJy).to(u.ABmag)
                    
                    if np.isfinite(mag.value):
                        # Vary magnitude relative to host
                        mag_value = random.uniform(mag.value - 1, mag.value + 3)
                    else:
                        try:
                            sampled_mags = self.mag_dist.sample(band, 1)
                            mag_value = float(sampled_mags[0]) if sampled_mags is not None else 22.0
                        except Exception:
                            mag_value = 22.0
                except Exception as e:
                    print(f"Warning: magnitude calculation failed: {e}")
                    mag_value = 22.0  # default magnitude
                
                mags.append(mag_value)
            
            return inject_ra, inject_dec, mags
            
        except Exception as e:
            print(f"Error in generate_positions_vectorized: {e}")
            # Return empty lists on critical error
            return [], [], []

    def generate_hostless_injections(self, n_hostless: int, band: str, image_shape: Tuple[int, int] = (2100, 4200)) -> Tuple[List[float], List[float], List[float]]:
        """
        Generate hostless injections distributed randomly across the image.
        
        Args:
            n_hostless (int): Number of hostless injections to generate
            band (str): Photometric band for magnitude sampling
            image_shape (Tuple[int, int]): Image dimensions (height, width) in pixels
            
        Returns:
            Tuple[List[float], List[float], List[float]]: 
                (x_pixels, y_pixels, magnitudes) for hostless injections
                
        Note:
            Positions are in pixel coordinates and need WCS conversion for RA/Dec.
            Returns empty lists on error.
        """
        try:
            # Generate random pixel positions across image
            x_values = np.random.uniform(0, image_shape[0], n_hostless)
            y_values = np.random.uniform(0, image_shape[1], n_hostless)
            
            # Sample magnitudes from distribution
            try:
                mag_values = self.mag_dist.sample(band, n_hostless)
                if mag_values is None:
                    mag_values = [22.0] * n_hostless  # default magnitudes
                else:
                    # Ensure proper list format
                    mag_values = [float(m) for m in mag_values]
            except Exception as e:
                print(f"Warning: magnitude sampling failed: {e}")
                mag_values = [22.0] * n_hostless
            
            return x_values.tolist(), y_values.tolist(), mag_values
            
        except Exception as e:
            print(f"Error in generate_hostless_injections: {e}")
            return [], [], []

class CatalogProcessor:
    """
    Handles catalog creation and processing for injection workflows.
    
    This class coordinates the generation of injection catalogs, combining
    galaxy-hosted and hostless sources, and provides utilities for catalog
    manipulation and visualization.
    
    Attributes:
        injection_gen (InjectionGenerator): Injection parameter generator
    """
    
    def __init__(self):
        """Initialize catalog processor with default injection generator."""
        self.injection_gen = InjectionGenerator()
        
    def process_catalog(self, catalog_of_galaxies: pd.DataFrame, image: Any, band: str, 
                    plot: bool = False, additional_host_data: bool = False,
                    config: Optional[InjectionConfig] = None) -> Tuple:
        """
        Process galaxy catalog to generate injection coordinates and magnitudes.
        
        This is the main method that coordinates injection generation, combining
        galaxy-hosted and hostless sources into a complete injection catalog.
        
        Args:
            catalog_of_galaxies (pd.DataFrame): Input galaxy catalog
            image (Any): LSST exposure object with WCS information
            band (str): Photometric band ('g', 'r', 'i', 'z', 'y')
            plot (bool): Whether to generate visualization plots
            additional_host_data (bool): Whether to include detailed host information
            config (Optional[InjectionConfig]): Configuration parameters
            
        Returns:
            Tuple: Either (ra, dec, mags) or (ra, dec, mags, host_data_df)
                depending on additional_host_data parameter
                
        Raises:
            Exception: Re-raises critical errors after logging
            
        Note:
            Combines galaxy-hosted injections with hostless background sources.
            Handles coordinate system conversions and error recovery gracefully.
        """
        
        try:
            # Generate hosted injections using vectorized approach
            inject_ra, inject_dec, mags = self.injection_gen.generate_positions_vectorized(
                catalog_of_galaxies, band
            )
            
            # Generate hostless injections 
            n_hostless = max(1, int(len(catalog_of_galaxies) * config.hostless_fraction))
            x_hostless, y_hostless, mag_hostless = self.injection_gen.generate_hostless_injections(
                n_hostless, band
            )
            
            # Convert hostless pixel positions to RA/Dec coordinates
            try:
                # Get WCS from image (handle different LSST stack versions)
                wcs = image.wcs if hasattr(image, 'wcs') else image.getWcs()
                ra_hostless_list = []
                dec_hostless_list = []
                
                for x, y in zip(x_hostless, y_hostless):
                    try:
                        # Convert pixel to sky coordinates
                        sky_coord = wcs.pixelToSky(lsst.geom.Point2D(x, y))
                        ra_hostless_list.append(sky_coord.getRa().asDegrees())
                        dec_hostless_list.append(sky_coord.getDec().asDegrees())
                    except Exception as e:
                        print(f"Warning: coordinate conversion failed for pixel ({x}, {y}): {e}")
                        continue
                
                ra_hostless = np.array(ra_hostless_list)
                dec_hostless = np.array(dec_hostless_list)
                # Trim magnitude list to match successful conversions
                mag_hostless = mag_hostless[:len(ra_hostless_list)]
                
            except Exception as e:
                print(f"Error in coordinate conversion: {e}")
                # Fallback: use approximate pixel-to-sky conversion
                ra_hostless = np.array(x_hostless) * 0.168432 / 3600  # approximate
                dec_hostless = np.array(y_hostless) * 0.168432 / 3600
            
            # Combine hosted and hostless injections
            inject_ra.extend(ra_hostless.tolist())
            inject_dec.extend(dec_hostless.tolist())
            mags.extend(mag_hostless)
            
            # Create detailed host data if requested
            host_data = None
            if additional_host_data:
                host_data = self._create_host_data(
                    catalog_of_galaxies, inject_ra[:len(catalog_of_galaxies)], 
                    inject_dec[:len(catalog_of_galaxies)], mags[:len(catalog_of_galaxies)],
                    ra_hostless, dec_hostless, mag_hostless, band
                )
            
            # Generate diagnostic plots if requested
            if plot:
                try:
                    self._plot_injections(image, catalog_of_galaxies, inject_ra, inject_dec, x_hostless, y_hostless)
                except Exception as e:
                    print(f"Warning: plotting failed: {e}")
            
            return (inject_ra, inject_dec, mags, host_data) if additional_host_data else (inject_ra, inject_dec, mags)
            
        except Exception as e:
            print(f"Error in process_catalog: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _create_host_data(self, catalog_of_galaxies: pd.DataFrame, inject_ra: List[float], 
                         inject_dec: List[float], mags: List[float], ra_hostless: np.ndarray,
                         dec_hostless: np.ndarray, mag_hostless: List[float], band: str) -> pd.DataFrame:
        """
        Create comprehensive host data DataFrame with injection and galaxy information.
        
        Args:
            catalog_of_galaxies (pd.DataFrame): Original galaxy catalog
            inject_ra (List[float]): RA coordinates of hosted injections
            inject_dec (List[float]): Dec coordinates of hosted injections  
            mags (List[float]): Magnitudes of hosted injections
            ra_hostless (np.ndarray): RA coordinates of hostless injections
            dec_hostless (np.ndarray): Dec coordinates of hostless injections
            mag_hostless (List[float]): Magnitudes of hostless injections
            band (str): Photometric band
            
        Returns:
            pd.DataFrame: Combined catalog with injection and host information
            
        Note:
            Includes metadata like visit/detector IDs and parent galaxy indices.
            Handles missing data gracefully with None values.
        """
        
        try:
            # Calculate host galaxy magnitudes from flux measurements
            host_magnitudes = []
            for flux in catalog_of_galaxies[DEFAULT_FLUX_TYPE]:
                try:
                    mag_val = (flux * u.nJy).to(u.ABmag).value
                    host_magnitudes.append(mag_val if np.isfinite(mag_val) else None)
                except Exception:
                    host_magnitudes.append(None)
            
            # Extract visit and detector IDs if available
            visit_id = None
            detector_id = None
            
            if 'visit' in catalog_of_galaxies.columns:
                visit_id = catalog_of_galaxies['visit'].iloc[0]
            if 'detector' in catalog_of_galaxies.columns:
                detector_id = catalog_of_galaxies['detector'].iloc[0]
            
            # Create data structure for hosted injections
            n_hosted = len(catalog_of_galaxies)
            hosted_data = {
                'ra': inject_ra,
                'dec': inject_dec,
                'mag': mags,
                'host_magnitude': host_magnitudes,
                'visit': [visit_id] * n_hosted,  
                'detector': [detector_id] * n_hosted,  
                'parent_index': list(range(n_hosted)),  # Index into original catalog
                'band': [band] * n_hosted 
            }
            
            # Create data structure for hostless injections
            n_hostless = len(ra_hostless)
            hostless_data = {
                'ra': ra_hostless.tolist(),
                'dec': dec_hostless.tolist(),
                'mag': mag_hostless,
                'host_magnitude': [None] * n_hostless,  # No host galaxy
                'visit': [visit_id] * n_hostless,
                'detector': [detector_id] * n_hostless,
                'parent_index': [None] * n_hostless,  # No parent galaxy
                'band': [band] * n_hostless
            }
            
            # Combine hosted and hostless data
            all_data = {}
            for key in hosted_data.keys():
                all_data[key] = hosted_data[key] + hostless_data[key]
            
            return pd.DataFrame(all_data)
            
        except Exception as e:
            print(f"Error creating host data: {e}")
            # Return minimal DataFrame on error
            return pd.DataFrame({
                'ra': inject_ra + ra_hostless.tolist(),
                'dec': inject_dec + dec_hostless.tolist(),
                'mag': mags + mag_hostless,
                'band': [band] * (len(inject_ra) + len(ra_hostless))
            })

    def _plot_injections(self, image: Any, catalog_of_galaxies: pd.DataFrame, 
                        inject_ra: List[float], inject_dec: List[float],
                        x_hostless: List[float], y_hostless: List[float]):
        """
        Create diagnostic plots showing injection positions overlaid on image.
        
        Args:
            image (Any): LSST exposure object
            catalog_of_galaxies (pd.DataFrame): Galaxy catalog  
            inject_ra (List[float]): RA coordinates of hosted injections
            inject_dec (List[float]): Dec coordinates of hosted injections
            x_hostless (List[float]): X pixel coordinates of hostless injections
            y_hostless (List[float]): Y pixel coordinates of hostless injections
            
        Note:
            Creates matplotlib figure with color-coded markers:
            - 'x' marks: hosted injections
            - 'o' marks: host galaxies (same color as their injection)
            - '*' marks: hostless injections
        """
        # Set up LSST display system
        afwDisplay.setDefaultBackend('matplotlib')
        fig = plt.figure(figsize=(10, 8))
        afw_display = afwDisplay.Display(1)
        afw_display.scale('asinh', 'zscale')  # Good scaling for astronomical images
        afw_display.mtv(image)
        
        wcs = image.getWcs()
        
        # Plot hosted injections and their host galaxies with matching colors
        with afw_display.Buffering():
            for i, (_, row) in enumerate(catalog_of_galaxies.iterrows()):
                if i < len(inject_ra):
                    color = self._random_hex_color()
                    
                    # Plot injection position
                    pixel_pos = wcs.skyToPixel(lsst.geom.SpherePoint(
                        inject_ra[i], inject_dec[i], lsst.geom.degrees
                    ))
                    afw_display.dot('x', pixel_pos.getX(), pixel_pos.getY(), size=100, ctype=color)
                    
                    # Plot host galaxy position
                    host_pos = wcs.skyToPixel(lsst.geom.SpherePoint(
                        row['ra'], row['dec'], lsst.geom.degrees
                    ))
                    afw_display.dot('o', host_pos.getX(), host_pos.getY(), size=100, ctype=color)
        
        # Plot hostless injections with random colors
        with afw_display.Buffering():
            for x, y in zip(x_hostless, y_hostless):
                color = self._random_hex_color()
                afw_display.dot('*', x, y, size=100, ctype=color)
        
        plt.show()
    
    @staticmethod
    def _random_hex_color() -> str:
        """
        Generate a random hex color string.
        
        Returns:
            str: Random color in hex format (e.g., '#FF5733')
        """
        return f"#{random.randint(0, 0xFFFFFF):06x}"

def create_catalog(dict_data: Dict[str, Any], host_data: bool = False) -> Table:
    """
    Create an Astropy Table catalog from dictionary data.
    
    This function converts injection data into the format required by
    LSST injection tasks, with optional host galaxy information.
    
    Args:
        dict_data (Dict[str, Any]): Dictionary containing injection data with keys:
            - 'ra': Right ascension coordinates
            - 'dec': Declination coordinates  
            - 'mag': Source magnitudes
            - Additional optional keys for host_data mode
        host_data (bool): Whether to include host galaxy metadata
        
    Returns:
        Table: Astropy Table ready for LSST injection pipeline
        
    Example:
        >>> data = {'ra': [150.0, 151.0], 'dec': [2.0, 2.1], 'mag': [22.0, 21.5]}
        >>> catalog = create_catalog(data)
    """
    # Base columns required for injection
    base_data = {
        "injection_id": np.arange(len(dict_data["ra"])),  # Unique identifiers
        "ra": dict_data["ra"],
        "dec": dict_data["dec"],
        "source_type": "Star",  # LSST injection expects point sources
        "mag": dict_data["mag"]
    }
    
    # Add host-related metadata if requested
    if host_data:
        base_data.update({
            "host_magnitude": dict_data["host_magnitude"],
            "visit": dict_data["visit"],
            "detector": dict_data["detector"],
            "parent_index": dict_data["parent_index"],
            "band": dict_data["band"]
        })
    
    # Convert to DataFrame then Astropy Table
    df = pd.DataFrame(base_data)
    return Table.from_pandas(df)

def inject_sources(input_exp: Any, visit_summary: Any, catalog_of_injection: Any, plot: bool = False) -> Tuple:
    """
    Inject fake sources into an LSST exposure using the injection pipeline.
    
    This function uses the LSST source injection framework to add artificial
    sources to an image, handling PSF convolution and proper flux scaling.
    
    Args:
        input_exp (Any): Input LSST exposure object
        visit_summary (Any): Visit summary with calibration information
        catalog_of_injection (Any): Astropy Table with injection coordinates
        plot (bool): Whether to display the result with injected sources highlighted
        
    Returns:
        Tuple: (injected_exposure, injection_catalog)
            - injected_exposure: Modified exposure with added sources
            - injection_catalog: Catalog of successfully injected sources
            
    Note:
        Uses LSST's VisitInjectTask which handles:
        - PSF convolution of injected sources
        - Proper flux calibration
        - Mask plane updates (INJECTED mask)
        - Catalog coordinate transformations
    """
    # Extract calibration information from visit summary
    detector_summary = visit_summary
    psf = detector_summary.getPsf()              # Point Spread Function
    photo_calib = detector_summary.getPhotoCalib()  # Photometric calibration
    wcs = detector_summary.getWcs()              # World Coordinate System
    
    # Configure injection task
    inject_config = VisitInjectConfig(process_all_data_ids=False)
    inject_task = VisitInjectTask(config=inject_config)
    
    # Perform the injection
    injected_output = inject_task.run(
        injection_catalogs=catalog_of_injection,
        input_exposure=input_exp.clone(),  # Work on a clone of the input exposure
        psf=psf,
        photo_calib=photo_calib,
        wcs=wcs,
    )
    
    # Optional visualization of results
    if plot:
        afwDisplay.setDefaultBackend('matplotlib')
        fig = plt.figure(figsize=(10, 8))
        afw_display = afwDisplay.Display(1)
        afw_display.setMaskTransparency(100)  # Make most masks transparent
        afw_display.setMaskTransparency(50, name="INJECTED")  # Highlight injected regions
        afw_display.scale('asinh', 'zscale')
        afw_display.mtv(injected_output.output_exposure)
        plt.show()
    
    return injected_output.output_exposure, injected_output.output_catalog

def process_all_ccds(datasetRefs: List[Any], butler: Any, band: str, 
                    config: Optional[InjectionConfig] = None,
                    csv: bool = False, save_filename: Optional[str] = None) -> pd.DataFrame:
    """
    Process multiple CCDs to generate injection catalogs across a full survey.
    
    This function processes multiple detector/visit combinations in batch,
    generating consistent injection catalogs across large datasets with
    proper error handling and progress reporting.
    
    Args:
        datasetRefs (List[Any]): List of LSST data references to process
        butler (Any): LSST data butler for data access
        band (str): Photometric band to process ('g', 'r', 'i', 'z', 'y')
        config (Optional[InjectionConfig]): Configuration parameters
        csv (bool): Whether to save results to CSV file
        save_filename (Optional[str]): Output filename (without extension)
        
    Returns:
        pd.DataFrame: Combined catalog of all injections with metadata
        
    Note:
        Processes each CCD independently with comprehensive error handling.
        Filters galaxies (extendedness == 1) and samples appropriately.
        Handles coordinate column naming variations across LSST data releases.
        
    Example:
        >>> config = InjectionConfig(min_injections=100, galaxy_fraction=0.8)
        >>> catalog = process_all_ccds(refs, butler, 'r', config=config, csv=True)
    """
    if config is None:
        config = InjectionConfig()
        
    processor = CatalogProcessor()
    all_catalogs = []
    
    # Process each dataset reference
    for reference in datasetRefs:
        ref = reference.dataId
        
        # Skip if not matching requested band
        if ref['band'] != band:
            continue
            
        try:
            # Load source detection catalog and calibrated exposure
            print(f"Attempting to load data for {ref}")
            detecs = butler.get('sourceTable', dataId=ref)
            calexp = butler.get('calexp', dataId=ref)
            
            # Convert to pandas DataFrame if needed
            if hasattr(detecs, 'to_pandas'):
                detecs_df = detecs.to_pandas()
            else:
                detecs_df = detecs
            
            # Handle coordinate column naming variations
            ra_col = None
            dec_col = None
            for col in detecs_df.columns:
                if 'ra' in col.lower():
                    ra_col = col
                if 'dec' in col.lower():
                    dec_col = col
            
            if ra_col is None or dec_col is None:
                print(f"Error: Could not find RA/Dec columns in {ref}")
                continue
            
            # Standardize column names
            if ra_col != 'ra':
                detecs_df = detecs_df.rename(columns={ra_col: 'ra'})
            if dec_col != 'dec':
                detecs_df = detecs_df.rename(columns={dec_col: 'dec'})
            
            # Filter for extended sources (galaxies) and sample based on galaxy_fraction
            filtered_detecs = detecs_df[detecs_df['extendedness'] == 1]
            
            # Calculate number of galaxies to keep based on galaxy_fraction
            nbr_fake = max(config.min_injections, int(len(filtered_detecs) * config.galaxy_fraction))
            
            if config.max_injections:
                nbr_fake = min(nbr_fake, config.max_injections)
            
            if nbr_fake == 0:
                print(f"Skipping {ref}: no galaxies found")
                continue
            
            # Adjust for available data
            if len(filtered_detecs) < nbr_fake:
                nbr_fake = len(filtered_detecs)
                print(f"Adjusting number of injections to {nbr_fake} (limited by available galaxies)")
                
            # Sample galaxies for injection hosts
            try:
                sampled_detecs = filtered_detecs.sample(n=nbr_fake, random_state=config.random_seed)
            except Exception as e:
                print(f"Error in sampling: {e}")
                # Fallback sampling method
                indices = np.random.RandomState(config.random_seed).choice(
                    len(filtered_detecs), size=nbr_fake, replace=False
                )
                sampled_detecs = filtered_detecs.iloc[indices]
            
            # Add metadata to sampled catalog
            sampled_detecs = sampled_detecs.copy()
            sampled_detecs['visit'] = ref['visit']
            sampled_detecs['detector'] = ref['detector']
            
            # Calculate data quality metrics
            try:
                finite_mags = []
                for flux in sampled_detecs[DEFAULT_FLUX_TYPE]:
                    try:
                        mag_val = (flux * u.nJy).to(u.ABmag).value
                        finite_mags.append(np.isfinite(mag_val))
                    except Exception:
                        finite_mags.append(False)
                
                finite_mag_per = sum(finite_mags) / len(finite_mags) * 100 if finite_mags else 0
            except Exception as e:
                print(f"Error calculating magnitudes for {ref}: {e}")
                continue
            
            if finite_mag_per <= 0:
                print(f"Skipping {ref}: no valid magnitudes")
                continue
            
            print(f"Processing {ref}: {len(filtered_detecs)} galaxies, "
                  f"{nbr_fake} injections, {finite_mag_per:.1f}% valid mags")
            
            # Generate injection catalog
            try:
                ra, dec, mags, df = processor.process_catalog(
                    sampled_detecs, calexp, band, False, True, config
                )
                
                # Format output data
                catalog_data = {
                    'ra': ra,
                    'dec': dec, 
                    'mag': mags,
                    'host_magnitude': df['host_magnitude'].tolist() if 'host_magnitude' in df else [None] * len(ra),
                    'visit': df['visit'].tolist() if 'visit' in df else [ref['visit']] * len(ra),
                    'detector': df['detector'].tolist() if 'detector' in df else [ref['detector']] * len(ra),
                    'parent_index': df['parent_index'].tolist() if 'parent_index' in df else list(range(len(ra))),
                    'band': df['band'].tolist() if 'band' in df else [band] * len(ra)
                }
                
                # Create formatted catalog
                fancy_catalog = create_catalog(catalog_data, True).to_pandas()
                all_catalogs.append(fancy_catalog)
                
            except Exception as e:
                print(f"Error in catalog processing for {ref}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        except ImportError as e:
            if 'numpy._core' in str(e):
                print(f"Numpy compatibility error for {ref}: {e}")
                print("Try updating numpy or using a compatible environment")
            else:
                print(f"Import error processing {ref}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {ref}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all processed catalogs
    if all_catalogs:
        result = pd.concat(all_catalogs, ignore_index=True)
        
        # Save to CSV if requested
        if csv and save_filename:
            # Use save_filename as-is (it should be the full path from config)
            output_path = f'{save_filename}.csv'
            try:
                # Create directory if it doesn't exist
                import os
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                result.to_csv(output_path, index=False)
                print(f"Saved catalog to: {output_path}")
            except Exception as e:
                print(f"Error saving CSV: {e}")
        
        return result
    else:
        print("No catalogs were successfully processed")
        return pd.DataFrame()

# Backward compatibility functions
def sample_from_distribution(band: str, num_samples: int = 1):
    """
    Sample magnitudes from pre-computed distributions (backward compatibility).
    
    Args:
        band (str): Photometric band  
        num_samples (int): Number of samples to generate
        
    Returns:
        np.ndarray: Sampled magnitudes
    """
    mag_dist = MagnitudeDistribution()
    return mag_dist.sample(band, num_samples)

def calculate_orientation_and_extent(ixx: float, ixy: float, iyy: float):
    """
    Calculate galaxy orientation and extent (backward compatibility).
    
    Args:
        ixx, ixy, iyy (float): Second moment components
        
    Returns:
        Tuple[float, float, float]: (orientation, semi_major, semi_minor)
    """
    return GeometryCalculator.calculate_orientation_and_extent(ixx, ixy, iyy)

def pos_mag_4catalog(catalog_of_galaxies: pd.DataFrame, image: Any, band: str, 
                    plot: bool = False, additional_host_data: bool = False):
    """
    Generate positions and magnitudes for catalog (backward compatibility).
    
    Args:
        catalog_of_galaxies (pd.DataFrame): Galaxy catalog
        image (Any): LSST exposure
        band (str): Photometric band
        plot (bool): Generate plots
        additional_host_data (bool): Include host data
        
    Returns:
        Tuple: Injection coordinates and magnitudes
    """
    processor = CatalogProcessor()
    return processor.process_catalog(catalog_of_galaxies, image, band, plot, additional_host_data)