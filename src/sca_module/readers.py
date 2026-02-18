import numpy as np
import xarray as xr

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
    data = data.where(~(conditions_post))

    return data