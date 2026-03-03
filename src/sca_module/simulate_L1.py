import scipy
import numpy as np
import xarray as xr

from drama.performance.sar.antenna_patterns import sinc_bp, phased_array
from leakage.misc import calculate_distance
from leakage.utils import mean_along_azimuth
from leakage.low_pass_filter import low_pass_filter_2D_dataset
from leakage.conversions import dB, phase2vel, dop2vel, vel2dop, slant2ground
from leakage.frequency_domain_padding import padding_fourier, da_integer_oversample_like, compute_padding_1D
from leakage.uncertainties import pulse_pair_coherence, generate_complex_speckle, speckle_intensity, phase_error_generator, speckle_coherence, phase_error_generator_2D, generate_correlated_complex_speckle


def create_observation_cube(
        ds, 
        az_footprint_cutoff, 
        grid_spacing, 
        vx_sat, 
        PRF,
        dim_original = "grg",
        dim_filter = "az",
        dim_new = "az_slow_time",
        dim_window = "az_beam",
        dropna = True
        ):
    """
    Function to remove data outside beam pattern footprint.
    Creates a new stack of (potentially overlapping) observations centered on a new azimuthmul coordinate system

    NOTE does not work for squinted geometries
    """

    window_size = np.round(az_footprint_cutoff / grid_spacing).astype("int")
    stride = vx_sat / PRF
    stride_elements = np.round(vx_sat / PRF / grid_spacing).astype("int")

    data = ds.copy()

    data = data.chunk({dim_filter: 1, dim_original:"auto"})
    data = data.rolling({dim_filter: window_size}, center=True).construct({dim_filter:dim_window}, stride=stride_elements)
    
    stride = (data[dim_filter][-1] - data[dim_filter][0]) / (data[dim_filter].sizes[dim_filter]-1)
    stride = float(stride.data)
    
    slow_time = np.arange(data[dim_filter].sizes[dim_filter]) * stride + data[dim_filter][0].data
    
    data = data.assign_coords(
                    {dim_new: (dim_filter, slow_time),
                    dim_window: (data[dim_window] * grid_spacing)}
                )
    data = data.swap_dims({dim_filter: dim_new})
    data = data.reset_coords(names=dim_filter)

    data = data.assign_coords(
        {dim_window: (data[dim_window] - data[dim_window].mean())}
    )

    delta = 1E-10
    to_clip = np.round(window_size/stride_elements + delta, 0).astype(int)
    start_idx = np.round(to_clip / 2).astype(int)
    end_idx = (to_clip - start_idx).astype(int)
    data = data.isel({dim_new : slice(start_idx, -end_idx)})
    data = data.chunk({dim_new: "auto", dim_original:"auto", dim_window:-1})
    data.attrs["stride"] = stride
    data.attrs["window_size"] = window_size

    if dropna:
        data = data.dropna(dim = dim_new)

    return data

def compute_beam_pattern(az_angle_wrt_boresight: np.array, grg_angle_wrt_boresight: np.array, antenna_elements: int, antenna_weighting: float, antenna_length: float, antenna_height: float, f0: float, beam_pattern: str):
    """
    Computes a beam pattern to be applied along the entire dataset.
    # NOTE The following computations are directly computed, a delayed lazy computation may be better
    # NOTE Currently tapering only azimuth
    # NOTE Same beam pattern assumed for transmit and receive
    # NOTE Beam patterns are already in intensity, square is only needed for two-way pattern

    Input
    -----
    antenna_elements int; 
        number of anetenna elements that are considered in beam tapering, only affects when beam pattern = phased_array
    antenna_weighting: 
        float, int; weighting parameter as defined by the called beam pattern functions. only affects when beam pattern = phased_array
    """

    # Assumes same pattern on transmit and receive
    if beam_pattern == "sinc":
        beam_az = sinc_bp(
            sin_angle=np.sin(az_angle_wrt_boresight),
            L=antenna_length,
            f0=f0,
        )
        
    elif beam_pattern == "phased_array":
        beam_az = phased_array(
            sin_angle=np.sin(az_angle_wrt_boresight),
            L=antenna_length,
            f0=f0,
            N=antenna_elements,
            w=antenna_weighting,
        ).squeeze()

    beam_grg = sinc_bp(
        sin_angle=np.sin(grg_angle_wrt_boresight),
        L=antenna_height,
        f0=f0,
    )

    beam_az_two_way = beam_az**2
    beam_grg_two_way = beam_grg**2

    return beam_az_two_way * beam_grg_two_way

def add_L1_uncertainties(
        ds: xr.Dataset, 
        dim_range: str, 
        dim_az_beam: str, 
        vx_sat: float|int, 
        U: float|int, 
        sat_height: float|int, 
        T_pp: float|int, 
        Lambda: float, 
        grid_spacing_slant_range: float, 
        fixed_SCR: bool = None,
        var_reference: str = "V_leakage_pulse_rg", 
        interpolator: str = "linear", 
        random_state: int = 42,

        ):
    """
    Here we upsample the dataset so that we can add noise at the native resolution in slant range.
    - first the data is interpolated to a slant-range spacing defined by the resolution (e.g. 150 m slant range resolution is 75m slant range grid)
    - then pulse-pair noise and speckle are generated per resolution cell, i.e. one noise sample per two grid cells
    - using zero-padding in the Fourier domain this is interpolated to noise/speckle at a 75 m spacing

    :param ds: xarray Dataset containing the observation data with all variables and coordinates
    :type ds: xr.Dataset
    :param dim_range: name of the range dimension (typically 'grg' for ground range)
    :type dim_range: str
    :param dim_az_beam: name of the azimuth beam dimension
    :type dim_az_beam: str
    :param vx_sat: satellite velocity in m/s
    :type vx_sat: float | int
    :param U: surface wind speed in m/s used for correlation time computation
    :type U: float | int
    :param sat_height: satellite altitude/height in meters
    :type sat_height: float | int
    :param T_pp: pulse-pair separation time in seconds
    :type T_pp: float | int
    :param Lambda: radar wavelength in meters
    :type Lambda: float
    :param grid_spacing_slant_range: targetted grid spacing of observations in meters
    :type grid_spacing_slant_range: float
    :param fixed_SCR: value to be used for a fixed signal to clutter ratio, defaults to None (calculates SCR from the data)
    :type fixed_SCR: float | int
    :param var_reference: reference variable name for determining output shape, defaults to "V_leakage_pulse_rg"
    :type var_reference: str
    :param interpolator: interpolation method to use (e.g., 'linear'), defaults to "linear"
    :type interpolator: str
    :param random_state: random seed for reproducibility, defaults to 42
    :type random_state: int
    """

    # prevent overwriting
    data_cube = ds.copy()
    data_cube = data_cube.astype("float32")
    
    # fix random state
    np.random.seed(random_state)

    # ------------------------------------
    # ---- Interpolate to slant range ----
    # ------------------------------------
    # NOTE to do this nicely the azimuth angle should also be taken into account, as it slightly affects local incidence angle
    # NOTE slant range does not account for Earths curvature here
    new_grg_pixel = slant2ground(
        spacing_slant_range=grid_spacing_slant_range,
        height=sat_height,
        ground_range_max=data_cube[dim_range].max().data * 1,
        ground_range_min=data_cube[dim_range].min().data * 1,
    )

    # interpolate data to new grg range pixels (effectively a variable low pass filter)
    data_cube_resamp = data_cube.interp({dim_range:new_grg_pixel}, method=interpolator)

    # ---------------------------------------------
    # ---- prepare Fourier domain zero padding ----
    # ---------------------------------------------
    dim_to_pad = list(data_cube_resamp[var_reference].dims).index(dim_range)
    dim_to_pad_name = f"dim_{dim_to_pad}"
    shape_ref = data_cube_resamp[var_reference].shape

    # assumes data along range is oversampled by a factor 2, such that there are two samples per independent sample resolution
    da_ones_independent_slant_range = da_integer_oversample_like(
        data_cube_resamp[var_reference], dim_to_resample=dim_to_pad, new_samples_per_original_sample=2
    )

    # calculate padding needed for slant-range
    pad = compute_padding_1D(
        length_desired=data_cube_resamp[var_reference].sizes[dim_range],
        length_current=da_ones_independent_slant_range.sizes[dim_to_pad_name],
    )

    # ----------------------------
    # ---- Compute coehrences ----
    # ----------------------------
    # calculates average azimuthal beam standard deviation within -3 dB
    beam_db = dB(data_cube_resamp.beam)
    beam_3dB = (
        xr.where((beam_db - beam_db.max(dim=dim_az_beam)) < -3, np.nan, 1)
        * data_cube_resamp.az_angle_wrt_boresight # in radians
    )
    sigma_az_angle = beam_3dB.std(dim=dim_az_beam).mean().values * 1

    wavenumber = 2 * np.pi / Lambda
    T_corr_Doppler = 1 / (np.sqrt(2) * wavenumber * vx_sat * sigma_az_angle)  
    T_corr_surface = 3.29 * Lambda / U

    # ------------------------
    # ---- adding speckle ----
    # ------------------------
    gamma_speckle = speckle_coherence(
        T_pp=T_pp,
        T_corr_surface=T_corr_surface,
        T_corr_Doppler=T_corr_Doppler,
    )

    # compute speckle for the first pulse
    speckle_c = generate_complex_speckle(
        da_ones_independent_slant_range.shape, random_state=random_state
    )

    # speckle for the second pulse is correlated to the first, we must use a different random state for the uncorrelated part
    speckle_c_displaced = generate_correlated_complex_speckle(speckle_c, correlation=gamma_speckle, random_state=random_state+1)

    da_speckle_c = speckle_c * da_ones_independent_slant_range
    da_speckle_c_displaced = speckle_c_displaced * da_ones_independent_slant_range
    da_speckle_c_padded = padding_fourier(da_speckle_c, padding=(pad, pad), dimension=dim_to_pad_name)
    da_speckle_c_displaced_padded = padding_fourier(da_speckle_c_displaced, padding=(pad, pad), dimension=dim_to_pad_name)

    # since iid noise, we can clip time domain to correct dimensions without affecting statistics
    shape_ref = data_cube_resamp[var_reference].shape
    da_speckle_c_padded_cut = da_speckle_c_padded[: shape_ref[0], : shape_ref[1]]
    da_speckle_c_displaced_padded_cut = da_speckle_c_displaced_padded[: shape_ref[0], : shape_ref[1]]

    speckle = speckle_intensity(da_speckle_c_padded_cut)
    speckle_displaced = speckle_intensity(da_speckle_c_displaced_padded_cut)
    nrcs_pulse_1 = data_cube_resamp['nrcs_scat'] * speckle.data
    nrcs_pulse_2 = data_cube_resamp['nrcs_scat'] * speckle_displaced.data

    # displace the second pulse down slant range and interpolate back to grid spacing
    # NOTE divided by two for two way travel time
    pulse_pair_slant_range_offset = T_pp * scipy.constants.c / 2
    nrcs_pulse_2 = nrcs_pulse_2.assign_coords(grg = data_cube_resamp.grg - pulse_pair_slant_range_offset).interp(grg = data_cube_resamp.grg)

    # add both pulses together
    # NOTE divided by two to keep power the same
    data_cube_resamp["nrcs_scat_speckle_pulse_1"] = nrcs_pulse_1 / 2
    data_cube_resamp["nrcs_scat_speckle_pulse_2"] = nrcs_pulse_2 / 2
    data_cube_resamp["nrcs_scat_speckle"] = (nrcs_pulse_1 + nrcs_pulse_2)/2

    # re-interpolate to higher sampling to maintain uniform ground samples
    data_cube_resamp = data_cube_resamp.astype("float32")

    # -------------------------------
    # ---- Pulse-pair uncertaity ----
    # -------------------------------
        # NOTE assumes no squint
    if fixed_SCR:
        gamma = pulse_pair_coherence(
            T_pp=T_pp,
            T_corr_surface=T_corr_surface,
            T_corr_Doppler=T_corr_Doppler,
            SNR=fixed_SCR
        )

        phase_uncertainty = phase_error_generator(
            gamma=gamma,
            n_samples=(da_ones_independent_slant_range.shape),
            random_state=random_state,
        )
    
    elif not fixed_SCR:

        temp_independent_slant_range = np.linspace(data_cube_resamp[dim_range].min(), data_cube_resamp[dim_range].max(), da_ones_independent_slant_range.sizes[dim_to_pad_name])
        SNR_2D = (data_cube_resamp.nrcs_scat_speckle_pulse_2 / data_cube_resamp.nrcs_scat_speckle_pulse_1).interp({dim_range:temp_independent_slant_range}, method=interpolator)

        gamma_2D = pulse_pair_coherence(
            T_pp=T_pp,
            T_corr_surface=T_corr_surface,
            T_corr_Doppler=T_corr_Doppler,
            SNR=SNR_2D
        )

        # here we use a different random state again, to avoid some artifical correlation with speckle
        phase_uncertainty = phase_error_generator_2D(gamma_2D.data, theta=0, random_state=random_state + 2, n_bins=1001)

    else: 
        raise ValueError("fixed_SCR must be either True or False")

    velocity_error = phase2vel(
        phase=phase_uncertainty, 
        wavenumber=wavenumber,
        T=T_pp
    )

    da_V_pp = velocity_error * da_ones_independent_slant_range

    # replace nans and infs with zero to avoid issues with padding, these will be ignored in the final product as they correspond to areas with no data
    da_V_pp = xr.where(np.isfinite(da_V_pp), x = da_V_pp, y = 0) 
    V_pp_c = padding_fourier(da_V_pp, padding=(pad, pad), dimension=dim_to_pad_name)
    V_pp = V_pp_c[: shape_ref[0], : shape_ref[1]]
    V_pp = V_pp.real # NOTE if complex part is not negligible this will slightly underestimate pulsepair noise

    # here we add pulse-pair uncertainty and convert it to surface projected velocity
    data_cube_resamp["V_pp"] = (
        xr.zeros_like(data_cube_resamp["V_leakage_pulse_rg"]) + V_pp.data
    ) / np.sin(np.radians(data_cube_resamp["inc_scat"])) # in radians

    # combine pulse-pair ucnertainty with leakage for total V_sigma
    data_cube_resamp["V_sigma"] = (
        data_cube_resamp["V_leakage_pulse_rg"] + data_cube_resamp["V_pp"]
    )

    return data_cube_resamp

def compute_cube_observation_geometry(ds_cube, dim_az_beam, dim_az_slow_time, dim_range, var_inc, sat_height, boresight_elevation_angle_scat):
    data_cube = ds_cube.copy()
    data_cube['distance_ground'] = calculate_distance(
        x=data_cube[dim_az_beam], y=data_cube[dim_range]
    )

    data_cube["distance_slant_range"] = calculate_distance(
            x=data_cube["distance_ground"], y=sat_height
        )

    data_cube["az_angle_wrt_boresight"] = np.arcsin(
        (data_cube[dim_az_beam]) / data_cube["distance_slant_range"]
        ) # NOTE arcsin instead of tan as distance_slant_range includes azimuthal angle

    data_cube["grg_angle_wrt_boresight"] = np.deg2rad(
        data_cube[var_inc] - boresight_elevation_angle_scat
    ).isel({dim_az_slow_time : 0}) # NOTE each beam should be identical, so only one needed
    
    return data_cube

def compute_true_observations(ds_cube, var_nrcs, var_inc, dim_az_beam, vx_sat, Lambda, var_azimuth_angle_wrt_boresight: str = "az_angle_wrt_boresight"):
    data_cube = ds_cube.copy()

    data_cube["beam_weight"] = data_cube["beam"] / mean_along_azimuth(
        data_cube["beam"], 
        azimuth_dim = dim_az_beam
    ) 
    data_cube["weight"] = (
        data_cube["beam"] * data_cube[var_nrcs]
    ) / mean_along_azimuth(
        data_cube["beam"] * data_cube[var_nrcs],
        azimuth_dim = dim_az_beam
    )

    data_cube["inc_scat"] = mean_along_azimuth(
        data_cube[var_inc],
        azimuth_dim = dim_az_beam,
    )

    data_cube["dop_geom"] = vel2dop(
        velocity=vx_sat,
        Lambda=Lambda,
        angle_incidence=np.radians(data_cube[var_inc]),  # NOTE assumes flat Earth
        angle_azimuth=data_cube[var_azimuth_angle_wrt_boresight],
        degrees=False,
    )

    data_cube["dop_geom_beam_weighted"] = (
        data_cube["dop_geom"] * data_cube["weight"]
    ) 
    data_cube["dop_beam_weighted"] = (
        data_cube["dop"] * data_cube["weight"]
    )

    # beam and backscatter weighted geometric Doppler is interpreted as geophysical Doppler, i.e. Leakage
    data_cube["V_leakage"] = dop2vel(
        Doppler=data_cube["dop_geom_beam_weighted"],
        Lambda=Lambda,
        angle_incidence=np.radians(data_cube[var_inc]),
        angle_azimuth=np.pi
        / 2,  # the geometric doppler is interpreted as radial motion, so azimuth angle component must result in value of 1 (i.e. pi/2)
        degrees=False,
    )

    data_cube["V"] = dop2vel(
        Doppler=data_cube["dop_beam_weighted"],
        Lambda=Lambda,
        angle_incidence=np.radians(data_cube[var_inc]),
        angle_azimuth=np.pi
        / 2,  # the geometric doppler is interpreted as radial motion, so azimuth angle component must result in value of 1 (i.e. pi/2)
        degrees=False,
    )

    # calculate scatterometer nrcs at scatterometer resolution (integrate over beam)
    data_cube["nrcs_scat"] = mean_along_azimuth(
        data_cube[var_nrcs] * data_cube["beam_weight"],
        azimuth_dim = dim_az_beam
    )

    # integrate Doppler and leakage over beam
    data_cube[["dop_geom_pulse_rg", "V_leakage_pulse_rg", "dop_pulse_rg", "V_pulse_rg"]] = mean_along_azimuth(
        data_cube[["dop_geom_beam_weighted", "V_leakage", "dop_beam_weighted", "V"]], 
        azimuth_dim = dim_az_beam
    )
    return data_cube

def compute_lpf_cube(
        ds_cube, 
        fs_y,
        fs_x,
        resolution_product,
        data_2_lpf,
        window = 'hann',
        crop_edges = False,
    ):

    data_lpf = [name + "_lpf" for name in data_2_lpf]

    ds = ds_cube.copy()
    # we have to make sure that all arrays have the same dimension order prior to low-pass filtering
    ds[data_lpf] = low_pass_filter_2D_dataset(
        ds[data_2_lpf],
        cutoff_frequency=1 / (resolution_product),
        fs_x=fs_x,
        fs_y=fs_y,
        window=window,
        fill_nans=True,
        crop_edges=crop_edges,
    )
    return ds