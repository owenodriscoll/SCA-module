import numpy as np
import xarray as xr

R_EARTH = 6371e3

def create_dataset(nrcs = np.array, dop = np.array, inc = np.array, grg = np.array, az = np.array, grid_spacing=int, fill_nan_limit = 1):
    """
    Creates a new xarray dataset with the coordinates and dimensions of interest.

    Parameters
    ----------
    nrcs: np.array,
        nrcs, in dimensions (az, grg)
    dop: np.array,
        dop, in dimensions (az, grg)
    inc: np.array,
        incidence angle, in degrees, in dimensions (az, grg)
    grg: np.array,
        ground range, in dimensions (grg)
    az: np.array,
        azimuth range, in dimensions (az)
    grid_spacing: int,
        grid spacing in meters
    """
    dim_az = "az"
    dim_grg = "grg"

    # create new dataset
    data = xr.Dataset(
        data_vars=dict(
            nrcs=(
                [dim_az, dim_grg],
                nrcs,
                {"units": "m2/m2"},
            ),
            dop=(
                [dim_az, dim_grg],
                dop,
                {"units": "Hertz"},
            ),
            inc=(
                [dim_az, dim_grg],
                inc,
                {"units": "Degrees"},
            ),
        ),
        coords=dict(
            az=([dim_az], az, {"units": "m"}),
            grg=([dim_grg], grg, {"units": "m"}),
        ),
        attrs=dict(grid_spacing=grid_spacing),
    )

    condition_to_fix = (data["nrcs"].isnull()) | (data["nrcs"] <= 0)
    data["nrcs"] = data["nrcs"].where(~condition_to_fix)

    # fill nans using limit, limit = 1 fills only single missing pixels,limit = None fill all, limit = 0 filters nothing, not consistent missing data
    interpolater = lambda x: x.interpolate_na(
        dim=dim_az,
        method="linear",
        limit=fill_nan_limit,
        fill_value="extrapolate",
    )
    data["nrcs"] = interpolater(data["nrcs"])
    conditions_post = (data["nrcs"].isnull()) | (data["nrcs"] <= 0)
    data["nrcs"] = data["nrcs"].where(~(conditions_post))

    return data

def decompress_polynomial_fields(ds, variables = None, dim_sample = "sample", dim_coeffs = "coeffs"):
    """
    Evaluates fields that are stored as one polynomial along `sample` per line, as in the 1 km Sentinel-1 GRD files of DATA-2
    (listed in the attribute `vars_polynomial_compress`). Coefficients are ordered highest order first, as expected by np.polyval.

    NOTE azimuth_time is stored as float32 nanoseconds since 1970, i.e. with a resolution of ~100 s, and is unusable as time

    Parameters
    ----------
    ds: xr.Dataset,
        GRD dataset with the compressed fields in dimensions (line, coeffs)
    variables: list[str],
        fields to evaluate, defaults to those listed in `vars_polynomial_compress`

    Returns
    -------
    ds: xr.Dataset,
        copy of the input with the fields evaluated in dimensions (line, sample)
    """
    ds = ds.copy()
    if variables is None:
        variables = ds.attrs["vars_polynomial_compress"].split(",")

    sample = ds[dim_sample].data.astype("float64")
    for name in variables:
        coeffs = ds[name].transpose(..., dim_coeffs).data.astype("float64")
        values = np.stack([np.polyval(c, sample) for c in coeffs])
        ds[name] = (ds[name].dims[:-1] + (dim_sample,), values, ds[name].attrs)

    return ds


def stitch_s1_grd(files, pol = "VV", spacing = 1000, pixel_spacing = 10):
    """
    Stitches consecutive Sentinel-1 GRD slices along azimuth and resamples them onto a uniform azimuth grid.

    The along-track position of each line is the cumulative great-circle distance between the mid-swath positions of successive lines,
    which keeps the (sub-km) gaps between slices. Ground range is sample * pixel_spacing from the near-range edge.
    NOTE the slant range time of the first sample is the same for all slices of a datatake (to < 1e-7 s), so samples are aligned between slices
    NOTE only the samples shared by all slices are kept

    Parameters
    ----------
    files: list[str],
        GRD files of consecutive slices, in order of acquisition
    pol: str,
        polarisation of sigma0 and nesz to keep
    spacing: int,
        azimuth spacing of the output grid, in meters
    pixel_spacing: int,
        ground range spacing of one GRD sample unit, in meters (10 m for IW GRDH)

    Returns
    -------
    data: xr.Dataset,
        sigma0, nesz (both linear), incidence (degrees), latitude, longitude, ground_heading and land_mask,
        in dimensions (az, x) with az the along-track and x the ground-range distance, in meters
    """
    slices = [decompress_polynomial_fields(xr.open_dataset(file)) for file in files]
    n_sample = min(s.sizes["sample"] for s in slices)
    variables = ["sigma0", "nesz", "incidence", "latitude", "longitude", "ground_heading", "land_mask"]

    data = xr.concat(
        [s.sel(pol=pol).isel(sample=slice(0, n_sample))[variables].drop_vars("line") for s in slices],
        dim="line",
    ).astype("float64")

    # along-track distance between the mid-swath positions of successive lines
    lat = np.radians(data["latitude"].isel(sample=n_sample // 2).data)
    lon = np.radians(data["longitude"].isel(sample=n_sample // 2).data)
    hav = np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
    az = np.concatenate([[0], np.cumsum(2 * R_EARTH * np.arcsin(np.sqrt(hav)))])

    data = data.assign_coords(az=("line", az), x=("sample", data["sample"].data * pixel_spacing)).swap_dims(line="az", sample="x")
    data = data.drop_vars(["line", "sample"], errors="ignore")
    data = data.interp(az=np.arange(0, az[-1], spacing), method="linear")
    data["land_mask"] = data["land_mask"] > 0

    return data
