"""
DopSCA Configuration and Constants

This module provides a centralized configuration class for DopSCA simulations,
allowing notebooks to import consistent parameters from a single source.
"""

from typing import Optional
import numpy as np
import scipy.constants
import warnings


class DopSCAConfig:
    """Configuration class for DopSCA simulations containing system, pulse, product, antenna, and dimension parameters."""
    
    # -- System Details
    f0: float = 5.405e9  # Carrier frequency [Hz], Hoogeboom et al. (2018)
    z0: float = 823e3    # Satellite altitude [m], 823-848 km, Fois et al. (2015)
    vx_sat: int = 6800   # Satellite velocity [m/s], Hoogeboom et al. (2018)
    
    # -- Pulse Details
    PRF: int = 4                    # Pulse Repetition Frequency per antenna, Hoogeboom et al. (2018)
    SNR: float = 1.0                # Signal to noise ratio (~1 on average for Pulse Pair)
    T_pp: float = 1.15e-4           # Pulse-pair separation time [s]
    
    # -- Product Details
    boresight_elevation_angle_scat: float = 42      # Boresight elevation angle [deg]
    incidence_min: float = 30                       # Minium observation angle to consider
    grg_max: float = 850E3                          # Maximum ground range [m]
    grg_min: float = 550E3                          # Minimum ground range [m] (within range main lobe)
    grid_spacing_target: int = 340                  # Target grid spacing [m] (vx_sat / PRF / 5)
    grid_spacing_slant_range_SCA: float = 75        # Slant range resolution [m]
    az_footprint_cutoff: int = 80_000               # Azimuth footprint cutoff [m]
    resolution_product: int = 50_000                # Product resolution [m], Hoogeboom et al. (2018)
    product_averaging_window: str = "hann"          # Averaging window type
    
    # -- Antenna Details
    beam_pattern: str = "phased_array"       # Beam pattern type: "phased_array" or "sinc"
    antenna_length: float = 2.87             # Antenna length [m], mid beam, Fois et al. (2015)
    antenna_height: float = 0.32             # Antenna height [m], mid beam, Fois et al. (2015)
    antenna_elements: int = 4                # Number of antenna elements, mid beam, Rostan et al. (2016)
    antenna_weighting: float = 0.75          # Antenna weighting/tapering parameter
    
    # -- Dimensions (Dataset variable names)
    dim_az: str = 'az'
    dim_az_slow_time: str = 'az_slow_time'
    dim_az_beam: str = 'az_beam'
    dim_range: str = 'grg'
    var_nrcs: str = "nrcs"
    var_dop: str = "dop"
    var_inc: str = "inc"
    
    # -- Miscellaneous
    U: float = 6                                   # Average wind speed [m/s]
    fill_nan_limit: Optional[int] = 1              # NaN filling limit
    random_state: int = 42                         # Random seed for reproducibility
    interpolator: str = "linear"                   # Interpolation method
    
    @classmethod
    def get_wavelength(cls) -> float:
        """Calculate wavelength from carrier frequency."""
        return scipy.constants.c / cls.f0
    
    @classmethod
    def get_magic_offset(cls, incidence_min) -> float:
        """Calculate range offset to align with scatterometer observation geometry from the starting incidence angle (in radians)."""
        return cls.z0 * np.tan(incidence_min)
    
    @classmethod
    def get_grid_spacing_target(cls, magic_divider) -> int:
        """
        Calculate some finer grid spacing which to prevent aliasing due to, for instance, a low PRF. 
        Ideally the new gridspacing should be a multiple of the stallite velocity divided by the PRF, which is the distance between pulse centers.

        """
        new_grid_spacing = cls.vx_sat / cls.PRF / magic_divider
        if new_grid_spacing != int(new_grid_spacing):
            warnings.warn("New grid spacing is not an integer, which may cause issues with interpolation. Consider adjusting magic_divider to achieve an integer grid spacing.")
        return new_grid_spacing
    
    @classmethod
    def as_dict(cls) -> dict:
        """Return all configuration parameters as a dictionary."""
        return {
            # System details
            'f0': cls.f0,
            'z0': cls.z0,
            'vx_sat': cls.vx_sat,
            # Pulse details
            'PRF': cls.PRF,
            'SNR': cls.SNR,
            'T_pp': cls.T_pp,
            # Product details
            'boresight_elevation_angle_scat': cls.boresight_elevation_angle_scat,
            'grg_max': cls.grg_max,
            'grg_min': cls.grg_min,
            'grid_spacing_target': cls.grid_spacing_target,
            'grid_spacing_slant_range_SCA': cls.grid_spacing_slant_range_SCA,
            'az_footprint_cutoff': cls.az_footprint_cutoff,
            'resolution_product': cls.resolution_product,
            'product_averaging_window': cls.product_averaging_window,
            # Antenna details
            'beam_pattern': cls.beam_pattern,
            'antenna_length': cls.antenna_length,
            'antenna_height': cls.antenna_height,
            'antenna_elements': cls.antenna_elements,
            'antenna_weighting': cls.antenna_weighting,
            # Dimensions
            'dim_az': cls.dim_az,
            'dim_az_slow_time': cls.dim_az_slow_time,
            'dim_az_beam': cls.dim_az_beam,
            'dim_range': cls.dim_range,
            'var_nrcs': cls.var_nrcs,
            'var_dop': cls.var_dop,
            'var_inc': cls.var_inc,
            # Miscellaneous
            'U': cls.U,
            'fill_nan_limit': cls.fill_nan_limit,
            'random_state': cls.random_state,
            'interpolator': cls.interpolator,
        }
