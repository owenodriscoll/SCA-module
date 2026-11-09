import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

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


def interpolate_woc_rvl_to_grid(files, latitude, longitude, variables, angular_variables = (), max_distance = 1500):
    """
    Interpolates fields of the WOC Sentinel-1 RVL product (dimensions row, cell, swath) onto the positions of another grid.

    Each sub-swath is interpolated on its own (linear, on a Delaunay triangulation of the RVL cell centres of all files), and target
    points further than max_distance from that sub-swath's cell centres are left empty, so that the convex hull does not extend the sub-swath.
    Where sub-swaths overlap their values are averaged. Angular variables are interpolated as unit vectors.
    Fill values (-999, |x| > 1e30) are set to nan, also in latitude and longitude

    Parameters
    ----------
    files: list[str],
        RVL files, e.g. the consecutive files of one Sentinel-1 datatake
    latitude, longitude: xr.DataArray,
        positions of the target grid, in degrees
    variables: list[str],
        RVL variables to interpolate
    angular_variables: list[str],
        subset of variables that are angles in degrees (e.g. wind_direction, radial_direction)
    max_distance: float,
        largest distance between a target point and the nearest RVL cell centre, in meters

    Returns
    -------
    data: xr.Dataset,
        interpolated variables with the dimensions and coordinates of latitude
    """
    def clean(x):
        x = np.asarray(x, dtype="float64")
        return np.where((np.abs(x) > 1e30) | (x == -999), np.nan, x)

    rvl = [xr.open_dataset(file) for file in files]
    n_swath = rvl[0].sizes["swath"]

    # local equirectangular projection, in meters, shared by source and target
    lat0, lon0 = float(latitude.mean()), float(longitude.mean())
    def project(lat, lon):
        return np.stack([R_EARTH * np.radians(lon - lon0) * np.cos(np.radians(lat0)), R_EARTH * np.radians(lat - lat0)], axis=-1)
    target = project(latitude.data.ravel(), longitude.data.ravel())

    fields = {}
    for name in variables:
        if name in angular_variables:
            fields[name + "_cos"], fields[name + "_sin"] = [], []
        else:
            fields[name] = []

    for swath in range(n_swath):
        lat = np.concatenate([clean(d["lat"].isel(swath=swath)).ravel() for d in rvl])
        lon = np.concatenate([clean(d["lon"].isel(swath=swath)).ravel() for d in rvl])
        valid = np.isfinite(lat) & np.isfinite(lon)
        points = project(lat[valid], lon[valid])

        values = {}
        for name in variables:
            x = np.concatenate([clean(d[name].isel(swath=swath).squeeze()).ravel() for d in rvl])[valid]
            if name in angular_variables:
                values[name + "_cos"], values[name + "_sin"] = np.cos(np.radians(x)), np.sin(np.radians(x))
            else:
                values[name] = x

        interpolator = LinearNDInterpolator(points, np.stack(list(values.values()), axis=-1))
        result = interpolator(target)
        distance, _ = cKDTree(points).query(target)
        result[distance > max_distance] = np.nan
        for i, name in enumerate(values):
            fields[name].append(result[:, i])

    data = xr.Dataset(coords=latitude.coords)
    for name in variables:
        if name in angular_variables:
            c = np.nanmean(fields[name + "_cos"], axis=0)
            s = np.nanmean(fields[name + "_sin"], axis=0)
            values = np.degrees(np.arctan2(s, c)) % 360
        else:
            values = np.nanmean(fields[name], axis=0)
        data[name] = (latitude.dims, values.reshape(latitude.shape))

    return data
