import scipy
import numpy as np
import xarray as xr

from drama.performance.sar.antenna_patterns import sinc_bp, phased_array
from leakage.misc import calculate_distance
from leakage.utils import mean_along_azimuth
from leakage.low_pass_filter import low_pass_filter_2D_dataset
from leakage.conversions import phase2vel, dop2vel, vel2dop, slant2ground
from leakage.frequency_domain_padding import padding_fourier, da_integer_oversample_like, compute_padding_1D
from leakage.uncertainties import decorrelation, pulse_pair_coherence, generate_complex_speckle, speckle_intensity, speckle_coherence


def create_observation_cube(
        ds, 
        az_footprint_cutoff, 
        grid_spacing,
        v_sat_ground,
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
    The along-track spacing of the observations (stride) is v_sat_ground / PRF, with v_sat_ground the ground-track velocity

    NOTE does not work for squinted geometries
    """

    window_size = np.round(az_footprint_cutoff / grid_spacing).astype("int")
    stride = v_sat_ground / PRF
    stride_elements = np.round(v_sat_ground / PRF / grid_spacing).astype("int")

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

def compute_beam_pattern(az_angle_wrt_boresight_slant: np.array, grg_angle_wrt_boresight: np.array, antenna_elements: int, antenna_weighting: float, antenna_length: float, antenna_height: float, f0: float, beam_pattern: str):
    """
    Computes a beam pattern to be applied along the entire dataset.
    # NOTE The following computations are directly computed, a delayed lazy computation may be better
    # NOTE Currently tapering only azimuth
    # NOTE Same beam pattern assumed for transmit and receive
    # NOTE Beam patterns are already in intensity, square is only needed for two-way pattern

    Input
    -----
    az_angle_wrt_boresight_slant: np.array;
        azimuth angle w.r.t. boresight in the slant plane (antenna frame), in radians, i.e. arcsin(az / slant range)
    antenna_elements int; 
        number of anetenna elements that are considered in beam tapering, only affects when beam pattern = phased_array
    antenna_weighting: 
        float, int; weighting parameter as defined by the called beam pattern functions. only affects when beam pattern = phased_array
    """

    # Assumes same pattern on transmit and receive
    if beam_pattern == "sinc":
        beam_az = sinc_bp(
            sin_angle=np.sin(az_angle_wrt_boresight_slant),
            L=antenna_length,
            f0=f0,
        )
        
    elif beam_pattern == "phased_array":
        beam_az = phased_array(
            sin_angle=np.sin(az_angle_wrt_boresight_slant),
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
        v_sat_orbit: float|int,
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
    - the echoes of both pulses are simulated per resolution cell (same scatterers, decorrelated by surface and platform motion),
      overlapping with the echo of the other pulse from +- the pulse-pair offset. The received sample referenced to the second pulse
      gives the NRCS speckle, and the product of the two samples T_pp apart the pulse-pair noise
    - pulse-pair noise is returned as the real and imaginary part of that lag product, `pp_re` and `pp_im`, not as a velocity.
      Low-pass filter these and only then derive V_pp and V_sigma with combine_pulse_pair, so that the products are summed
      as by the processor (weighted by speckle and NRCS brightness, phases averaged on the circle)

    :param ds: xarray Dataset containing the observation data with all variables and coordinates
    :type ds: xr.Dataset
    :param dim_range: name of the range dimension (typically 'grg' for ground range)
    :type dim_range: str
    :param dim_az_beam: name of the azimuth beam dimension
    :type dim_az_beam: str
    :param v_sat_orbit: satellite velocity used for Doppler in m/s, sets the Doppler decorrelation time
    :type v_sat_orbit: float | int
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
    # ---- Compute coherences ----
    # ----------------------------
    # calculates average azimuthal beam standard deviation within -3 dB
    # NOTE samples where the beam is unknown (nan, e.g. beyond the extent of the beam pattern) count as outside the -3 dB beam
    beam = data_cube_resamp.beam
    beam_3dB = (
        xr.where(beam >= 10 ** (-3 / 10) * beam.max(dim=dim_az_beam), 1, np.nan)
        * data_cube_resamp.az_angle_wrt_boresight_slant # in radians, antenna-frame angle
    )
    sigma_az_angle = beam_3dB.std(dim=dim_az_beam).mean().values * 1
    # print(sigma_az_angle)

    wavenumber = 2 * np.pi / Lambda
    # NOTE assumes a single represenative sigma_az_angle over the entire range extent.
    # When simulating the full range-extent, a range-varying sigma_az_angle would likely be more suitable.
    T_corr_Doppler = 1 / (np.sqrt(2) * wavenumber * v_sat_orbit * sigma_az_angle)
    T_corr_surface = 3.29 * Lambda / U

    # -------------------------------------
    # ---- pulse-pair echo overlap ----
    # -------------------------------------
    # the echoes of both pulses overlap in the receiver: with time referenced to the first pulse, range gate r also
    # receives the echo of the second pulse from r - offset, and the sample T_pp later the echo of the first pulse from r + offset
    # NOTE divided by two for two way travel time
    pulse_pair_slant_range_offset = T_pp * scipy.constants.c / 2
    # data_cube_resamp is uniform in slant range, so the offset is an integer number of samples (rounding error ~10 m)
    offset_samples = int(np.round(pulse_pair_slant_range_offset / grid_spacing_slant_range))

    # NOTE the clutter is unknown where r -+ offset falls outside the scene: these cells are nan (one offset at either edge)
    nrcs_scat = data_cube_resamp['nrcs_scat']
    nrcs_scat_near = nrcs_scat.shift({dim_range: offset_samples})   # at r - offset
    nrcs_scat_far = nrcs_scat.shift({dim_range: -offset_samples})   # at r + offset

    # signal-to-clutter coherence from expected (speckle-free) powers: of the two samples T_pp apart only the echoes from r are
    # correlated, the clutter comes from r - offset in one and from r + offset in the other. Equals SCR / (1 + SCR) = 0.5 for uniform NRCS
    data_cube_resamp["gamma_SCR"] = nrcs_scat / np.sqrt((nrcs_scat + nrcs_scat_near) * (nrcs_scat + nrcs_scat_far))

    gamma_speckle = speckle_coherence(T_pp=T_pp, T_corr_surface=T_corr_surface, T_corr_Doppler=T_corr_Doppler)

    # ---------------------------------------------------
    # ---- echoes and received samples per resolution cell ----
    # ---------------------------------------------------
    if fixed_SCR:
        # uniform scene: unit-power speckle, coherence of the pair imposed directly
        speckle_c = generate_complex_speckle(da_ones_independent_slant_range.shape, random_state=random_state)
        gamma = pulse_pair_coherence(T_pp=T_pp, T_corr_surface=T_corr_surface, T_corr_Doppler=T_corr_Doppler, SNR=fixed_SCR)
        u_1, u_2 = generate_complex_speckle((2, *da_ones_independent_slant_range.shape), random_state=random_state + 2)
        sample_A, sample_B = u_1, gamma * u_1 + np.sqrt(1 - gamma**2) * u_2
    else:
        # expected power per resolution cell, positionally aligned with da_ones_independent_slant_range (same coarsening)
        P = (
            nrcs_scat.transpose(*data_cube_resamp[var_reference].dims)
            .coarsen({dim_range: 2}, boundary="trim")
            .mean()
            .data
        )
        P = np.moveaxis(P, dim_to_pad, -1)  # range last for shifting
        n = int(np.round(offset_samples / 2))

        def shift_range(x, k):
            """x at r - k along the last axis, nan where r - k falls outside the swath"""
            out = np.full_like(x, np.nan)
            if k > 0:
                out[..., k:] = x[..., :-k]
            else:
                out[..., :k] = x[..., -k:]
            return out

        # each pulse carries half the power; the second pulse sees the same scatterers, decorrelated by surface and platform motion
        u_echo, u_new = generate_complex_speckle((2, *P.shape), random_state=random_state)
        echo_1 = np.sqrt(P / 2) * u_echo
        echo_2 = np.sqrt(P / 2) * (gamma_speckle * u_echo + np.sqrt(1 - gamma_speckle**2) * u_new)

        # clutter: echo of the second pulse from r - offset and of the first pulse from r + offset
        # NOTE where r -+ offset falls outside the scene (or the scene is undefined) the clutter is unknown and the sample is nan, as gamma_SCR
        clutter_near = shift_range(echo_2, n)
        clutter_far = shift_range(echo_1, -n)
        P_far = shift_range(P, -n)

        # received samples T_pp apart, range referenced to the second pulse: sample B is also the detected NRCS sample
        sample_A = echo_1 + clutter_near
        sample_B = echo_2 + clutter_far

        # unit-power speckle of sample B, so that NRCS speckle and pulse-pair noise come from the same echoes
        speckle_c = sample_B / np.sqrt((P + P_far) / 2)
        speckle_c = np.where(np.isfinite(speckle_c), speckle_c, u_echo)  # nan cells: any unit speckle keeps the padding finite, the NRCS is nan there anyway
        speckle_c, sample_A, sample_B = [np.moveaxis(s, -1, dim_to_pad) for s in (speckle_c, sample_A, sample_B)]

    # ------------------------
    # ---- adding speckle ----
    # ------------------------
    da_speckle_c = speckle_c * da_ones_independent_slant_range
    da_speckle_c_padded = padding_fourier(da_speckle_c, padding=(pad, pad), dimension=dim_to_pad_name)

    # since iid noise, we can clip time domain to correct dimensions without affecting statistics
    shape_ref = data_cube_resamp[var_reference].shape
    da_speckle_c_padded_cut = da_speckle_c_padded[: shape_ref[0], : shape_ref[1]]
    speckle = speckle_intensity(da_speckle_c_padded_cut)

    # each pulse carries half the power. The overlapping echoes add as complex fields before detection,
    # so a single (single-look) speckle realisation applies to their combined power
    # range referenced to the second pulse: gate r holds the echo of the second pulse from r and of the first pulse from r + offset
    # NOTE assumes one detected sample per range gate.
    data_cube_resamp["nrcs_scat_speckle"] = (nrcs_scat + nrcs_scat_far) / 2 * speckle.data

    # re-interpolate to higher sampling to maintain uniform ground samples
    data_cube_resamp = data_cube_resamp.astype("float32")

    # -------------------------------
    # ---- Pulse-pair uncertaity ----
    # -------------------------------
    # the processor sums the complex lag-T_pp products and takes the phase of the sum. Each product is therefore weighted by
    # its magnitude (speckle and NRCS brightness), and phases are averaged on the circle (velocities alias at +- Lambda / (4 T_pp)).
    # The real and imaginary parts are stored; after low-pass filtering both, combine_pulse_pair takes the phase of the filtered product
    lag_product = np.conj(sample_A) * sample_B
    lag_product = {"pp_re": lag_product.real, "pp_im": lag_product.imag}
    for name, component in lag_product.items():
        # replace nans and infs with zero to avoid issues with padding, a zero product does not contribute to the sum
        da_component = xr.where(np.isfinite(component), component, 0) * da_ones_independent_slant_range
        component_c = padding_fourier(da_component, padding=(pad, pad), dimension=dim_to_pad_name)
        component_c = component_c[: shape_ref[0], : shape_ref[1]].real # NOTE if complex part is not negligible this will slightly underestimate pulsepair noise
        data_cube_resamp[name] = xr.zeros_like(data_cube_resamp[var_reference]) + component_c.data
        if not fixed_SCR:
            # nan where the clutter (or the scene) is unknown, as gamma_SCR
            data_cube_resamp[name] = data_cube_resamp[name].where(data_cube_resamp["gamma_SCR"].notnull())

    return data_cube_resamp

def compute_cube_observation_geometry(ds_cube, dim_az_beam, dim_az_slow_time, dim_range, var_inc, sat_height, boresight_elevation_angle_scat):
    data_cube = ds_cube.copy()
    data_cube['distance_ground'] = calculate_distance(
        x=data_cube[dim_az_beam], y=data_cube[dim_range]
    )

    data_cube["distance_slant_range"] = calculate_distance(
            x=data_cube["distance_ground"], y=sat_height
        )

    # azimuth angle in the slant plane (antenna frame): sin = az / slant range
    # use for the antenna beam pattern and the beam's angular spread (Doppler decorrelation time)
    # NOTE its sine already is the projection of the along-track direction onto the line of sight,
    # so it must not be combined with an additional sin(incidence) when computing Doppler
    data_cube["az_angle_wrt_boresight_slant"] = np.arcsin(
        (data_cube[dim_az_beam]) / data_cube["distance_slant_range"]
        ) # NOTE arcsin instead of tan as distance_slant_range includes azimuthal angle

    # azimuth angle in the ground plane: sin = az / ground distance
    # use with vel2dop, which itself projects from the ground plane onto the line of sight via sin(incidence)
    data_cube["az_angle_wrt_boresight_ground"] = np.arcsin(
        (data_cube[dim_az_beam]) / data_cube["distance_ground"]
        )

    data_cube["grg_angle_wrt_boresight"] = np.deg2rad(
        data_cube[var_inc] - boresight_elevation_angle_scat
    ).isel({dim_az_slow_time : 0}) # NOTE each beam should be identical, so only one needed
    
    return data_cube

def compute_true_observations(ds_cube, var_nrcs, var_inc, dim_az_beam, v_sat_orbit, Lambda, T_pp, var_azimuth_angle_wrt_boresight: str = "az_angle_wrt_boresight_ground"):
    """
    Integrates NRCS, Doppler and the geometric Doppler (leakage) over the azimuth beam.
    v_sat_orbit is the satellite velocity used for the geometric Doppler.

    The leakage is the phase of the beam-integrated lag product: each azimuth sample rotates its share of the lag product
    by its geometric Doppler phase over T_pp, and the pulse pair measures the phase of the sum. This differs from the
    power-weighted mean geometric Doppler by the odd higher-order cumulants (skewness) of the weighted Doppler spectrum

    NOTE where the azimuth window extends beyond the beam pattern the beam is unknown (nan). These samples are given zero gain,
    i.e. they are ignored in the beam integrals
    """
    data_cube = ds_cube.copy()
    data_cube["beam"] = data_cube["beam"].fillna(0)

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
        velocity=v_sat_orbit,
        Lambda=Lambda,
        angle_incidence=np.radians(data_cube[var_inc]),  # NOTE assumes flat Earth
        # NOTE vel2dop expects the ground-plane azimuth angle; passing the slant-plane angle applies sin(incidence) twice
        # NOTE var_inc is the zero-Doppler incidence, not the local incidence at each az_beam offset;
        # this leaves dop_geom up to ~0.2 % low at the footprint edge at near range, symmetric in azimuth
        angle_azimuth=data_cube[var_azimuth_angle_wrt_boresight],
        degrees=False,
    )

    data_cube["dop_beam_weighted"] = (
        data_cube["dop"] * data_cube["weight"]
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

    # integrate Doppler over beam
    data_cube[["dop_pulse_rg", "V_pulse_rg"]] = mean_along_azimuth(
        data_cube[["dop_beam_weighted", "V"]],
        azimuth_dim = dim_az_beam
    )

    # beam and backscatter weighted lag product of the geometric Doppler. Its phase is interpreted as geophysical Doppler, i.e. Leakage
    # NOTE phases beyond +- pi (|dop_geom| > 1 / (2 T_pp)) alias, as in the instrument
    lag_product_geom = mean_along_azimuth(
        data_cube["weight"] * np.exp(1j * 2 * np.pi * T_pp * data_cube["dop_geom"]),
        azimuth_dim = dim_az_beam
    )
    data_cube["dop_geom_pulse_rg"] = np.arctan2(lag_product_geom.imag, lag_product_geom.real) / (2 * np.pi * T_pp)
    data_cube["V_leakage_pulse_rg"] = dop2vel(
        Doppler=data_cube["dop_geom_pulse_rg"],
        Lambda=Lambda,
        angle_incidence=np.radians(data_cube["inc_scat"]),
        angle_azimuth=np.pi
        / 2,  # the geometric doppler is interpreted as radial motion, so azimuth angle component must result in value of 1 (i.e. pi/2)
        degrees=False,
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
        min_footprint_coverage = None,
    ):
    """
    Low-pass filters the variables in data_2_lpf to the product resolution, adding them as f"{name}_lpf".

    Nans are excluded by normalised convolution. With min_footprint_coverage, product cells are kept only where at least
    that fraction of the samples within their footprint is valid. The footprint is the filter's main lobe,
    +- resolution_product / 2 along both dimensions, and samples beyond the domain count as missing.
    min_footprint_coverage = 1 keeps only product cells whose whole footprint is available.
    NOTE fs_x is the sampling along the first dimension of the variables, fs_y along the second
    """

    ds = ds_cube.copy()
    kwargs = dict(
        cutoff_frequency=1 / (resolution_product),
        fs_x=fs_x,
        fs_y=fs_y,
        window=window,
        fill_nans=True,
        crop_edges=crop_edges,
    )
    # we have to make sure that all arrays have the same dimension order prior to low-pass filtering
    filtered = low_pass_filter_2D_dataset(ds[data_2_lpf], **kwargs)

    # normalised convolution: nans are filled with 0 before filtering, so the filtered valid-sample mask is divided out.
    # Averages then only run over valid samples, and nans (swath edges, land) no longer pull the result towards 0.
    # Without nans the weight is 1. Each distinct mask is filtered once
    masks, mask_of = {}, {}
    for name in data_2_lpf:
        valid = np.isfinite(ds[name]).astype("float32")
        mask_of[name] = next((key for key, mask in masks.items() if mask.equals(valid)), name)
        masks.setdefault(mask_of[name], valid)
    weights = low_pass_filter_2D_dataset(xr.Dataset(masks), **kwargs)

    # fraction of valid samples within each cell's footprint, as a separable box average. It is nan where the box leaves the domain
    # NOTE in float64: in float32 the rolling sums drift past many gaps, dropping fully covered cells just below the threshold
    complete = {key: True for key in masks}
    if min_footprint_coverage is not None:
        half_widths = [int(np.floor(resolution_product / 2 * fs)) for fs in (fs_x, fs_y)]
        for key, valid in masks.items():
            coverage = valid.astype("float64")
            for dim, half_width in zip(valid.dims, half_widths):
                coverage = coverage.rolling({dim: 2 * half_width + 1}, center=True).mean()
            complete[key] = coverage >= min_footprint_coverage - 1e-6

    for name in data_2_lpf:
        ds[name + "_lpf"] = (filtered[name] / weights[mask_of[name]]).where(complete[mask_of[name]])
    return ds

def combine_pulse_pair(
        ds,
        Lambda,
        T_pp,
        suffix = "",
        var_inc = "inc_scat",
    ):
    """
    Converts the pulse-pair lag product from add_L1_uncertainties into a surface-projected velocity error V_pp
    and adds it to the leakage to obtain V_sigma (and V_sigma_net if the net leakage is present).

    To be applied after low-pass filtering: the phase is taken from the filtered lag product, so that resolved samples are
    summed as complex products (as by a pulse-pair processor) rather than averaged as wrapped phases. Use suffix = "" for the native-resolution fields and
    suffix = "_lpf" for the product-resolution fields.

    Input
    -----
    ds: xr.Dataset,
        dataset containing f"pp_re{suffix}", f"pp_im{suffix}", f"V_leakage_pulse_rg{suffix}" and var_inc (in degrees),
        and optionally f"V_leakage_pulse_rg_net{suffix}"
    Lambda: float,
        radar wavelength, in meters
    T_pp: float,
        pulse-pair separation time, in seconds
    suffix: str,
        suffix of the fields to combine, e.g. "_lpf" for the low-pass filtered fields

    Returns
    -------
    ds: xr.Dataset,
        copy of the input with f"V_pp{suffix}", f"V_sigma{suffix}" and, if available, f"V_sigma_net{suffix}" added
    """
    ds = ds.copy()

    phase = np.arctan2(ds[f"pp_im{suffix}"], ds[f"pp_re{suffix}"])
    ds[f"V_pp{suffix}"] = phase2vel(
        phase=phase,
        wavenumber=2 * np.pi / Lambda,
        T=T_pp
    ) / np.sin(np.radians(ds[var_inc])) # surface projected

    ds[f"V_sigma{suffix}"] = ds[f"V_leakage_pulse_rg{suffix}"] + ds[f"V_pp{suffix}"]
    if f"V_leakage_pulse_rg_net{suffix}" in ds:
        ds[f"V_sigma_net{suffix}"] = ds[f"V_leakage_pulse_rg_net{suffix}"] + ds[f"V_pp{suffix}"]

    return ds

def multilook_nrcs_to_grid(
        ds,
        target,
        var_nrcs = "nrcs_scat_speckle",
        dim_range = "grg",
        dim_az_slow_time = "az_slow_time",
        dim_az = "az",
    ):
    """
    Brings the speckled scatterometer NRCS from its fine range sampling onto the coarser grid of `target`,
    e.g. to re-estimate leakage from SCA's own NRCS.

    The NRCS is averaged along range over one target grid cell before interpolation, as a ground processor would
    multi-look it, rather than point-sampling single-look speckle.
    NOTE no averaging along azimuth, as that would blur the azimuthal NRCS gradients that cause leakage

    Input
    -----
    ds: xr.Dataset,
        dataset containing var_nrcs on a uniform range grid, with dimensions (dim_az_slow_time, dim_range)
    target: xr.Dataset,
        dataset providing the target coordinates dim_az and dim_range, e.g. the output of readers.create_dataset

    Returns
    -------
    nrcs: xr.DataArray,
        range-averaged NRCS on the target grid, with dimensions (dim_az, dim_range)
    """
    spacing_range = float(ds[dim_range].diff(dim_range).median())
    spacing_target = float(target[dim_range].diff(dim_range).median())
    n_avg = max(int(np.round(spacing_target / spacing_range)), 1)

    nrcs = ds[var_nrcs].rolling({dim_range: n_avg}, center=True, min_periods=max(n_avg // 2, 1)).mean()
    nrcs = nrcs.interp({dim_az_slow_time: target[dim_az].data, dim_range: target[dim_range].data})

    return xr.DataArray(
        nrcs.transpose(dim_az_slow_time, dim_range).data,
        dims=(dim_az, dim_range),
        coords={dim_az: target[dim_az].data, dim_range: target[dim_range].data},
        name=var_nrcs,
    )