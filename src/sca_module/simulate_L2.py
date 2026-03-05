import numpy as np
import xarray as xr
from dataclasses import dataclass


NoneType = type(None)

def upsample_coords(ds, factor=2):
    new_coords = {
        dim: np.linspace(ds[dim].min().item(), ds[dim].max().item(), len(ds[dim]) * factor)
        for dim in ds.coords if ds.sizes[dim] > 1
    }
    return new_coords


@dataclass
class GMF_SWB:
    """
    Geophysical model functions as derived from Scientific Workbench Look Up Tables

    NOTE in default look-up table the parameter "inverse_wave_age" is erroneously called "wave_age" instead

    Parameters
    ----------
    LUT_Doppler: xr.Dataset
        dataset containing Doppler LUT from SWB
    LUT_nrcs: xr.Dataset
        dataset containing NRCS LUT from SWB
    supersample_factor: int
        factor by which to supersample input Look Up Tabels. Defaults to 4
    

    """
    LUT_Doppler: xr.Dataset
    LUT_nrcs: xr.Dataset
    supersample_factor: int = 4


    def __post_init__(self):
        self._upsample_data()

        return 

    def _upsample_data(self):
        if isinstance(self.supersample_factor, int):
            self.data_nrcs =  self.LUT_nrcs.interp(upsample_coords(self.LUT_nrcs, factor=self.supersample_factor))
            self.data_Doppler =  self.LUT_Doppler.interp(upsample_coords(self.LUT_Doppler, factor=self.supersample_factor))

        return


    @staticmethod
    def forward(data, v, phi, theta, inverse_wave_age=1):
        """
        Inputs should be given as (x,) array, all inputs should be the same size
        """

        wind_norm_ = xr.DataArray(v.ravel(), dims='points')
        wind_direction_ = xr.DataArray(phi.ravel(), dims='points')
        incidence_ = xr.DataArray(theta.ravel(), dims='points')

        if isinstance(inverse_wave_age, (int, float)):
            inverse_wave_age_ = xr.ones_like(incidence_) * inverse_wave_age
        else:
            inverse_wave_age_ = xr.DataArray(inverse_wave_age.ravel(), dims='points')

        # Check bounds for all coordinates
        v_min, v_max = float(data['wind_norm'].min()), float(data['wind_norm'].max())
        phi_min, phi_max = float(data['wind_direction'].min()), float(data['wind_direction'].max())
        theta_min, theta_max = float(data['incidence'].min()), float(data['incidence'].max())
        wa_min, wa_max = float(data['wave_age'].min()), float(data['wave_age'].max())

        # Create validity mask
        valid = (
            (wind_norm_ >= v_min) & (wind_norm_ <= v_max) &
            (wind_direction_ >= phi_min) & (wind_direction_ <= phi_max) &
            (incidence_ >= theta_min) & (incidence_ <= theta_max) &
            (inverse_wave_age_ >= wa_min) & (inverse_wave_age_ <= wa_max)
        )

        result = data.sel(
            wind_norm=wind_norm_, 
            wind_direction=wind_direction_, 
            incidence=incidence_, 
            wave_age=inverse_wave_age_,
            method='nearest',
        ).values.reshape(*v.shape)

        # Set out-of-bounds values to NaN
        result = result.astype(float)
        result[~valid.values.reshape(*v.shape)] = np.nan

        return result

    @staticmethod
    def inverse(data, y, phi, theta, inverse_wave_age=1):
        """
        Inputs should be given as (x,) array, all inputs should be the same size
        """

        wind_direction_ = xr.DataArray(phi.ravel(), dims='points')
        incidence_ = xr.DataArray(theta.ravel(), dims='points')

        if isinstance(inverse_wave_age, (int, float)):
            inverse_wave_age_ = xr.ones_like(incidence_) * inverse_wave_age
        else:
            inverse_wave_age_ = xr.DataArray(inverse_wave_age.ravel(), dims='points')

        # Check bounds for coordinate dimensions (excluding wind_norm which is the LUT dimension being searched)
        phi_min, phi_max = float(data['wind_direction'].min()), float(data['wind_direction'].max())
        theta_min, theta_max = float(data['incidence'].min()), float(data['incidence'].max())
        wa_min, wa_max = float(data['wave_age'].min()), float(data['wave_age'].max())

        # Create validity mask
        valid = (
            (wind_direction_ >= phi_min) & (wind_direction_ <= phi_max) &
            (incidence_ >= theta_min) & (incidence_ <= theta_max) &
            (inverse_wave_age_ >= wa_min) & (inverse_wave_age_ <= wa_max)
        )

        subset = data.sel(
            wind_direction=wind_direction_, 
            incidence=incidence_, 
            wave_age=inverse_wave_age_,
            method='nearest',
        )

        # -- compute differences to find minimum index
        abs_diff = np.abs(subset - y.ravel())
        min_idx = abs_diff.argmin(dim="wind_norm") 

        # -- extract corresponding wind_norm values
        most_likely_wind_norms = subset["wind_norm"].isel(wind_norm=min_idx)
        result = most_likely_wind_norms.data.reshape(*y.shape)

        # Set out-of-bounds values to NaN
        result = result.astype(float)
        result[~valid.values.reshape(*y.shape)] = np.nan

        return result
    

    def nrcs_forward(self, v, phi, theta, inverse_wave_age=1):
        """
        Estimates NRCS from wind speed

        Input
        -----
        v : array like
            wind speed in m/s
        phi : array like
            wind direction in degrees with respect to sensor (between 0-360), 0 & 360 is blowing away, 180 is blowing towards
        theta : 
            incidence angle in degrees 
        inverse_wave_age : int | float | array like
            wave age of system, default = 1

        Returns
        -------
        result : array_like
            estimated NRCS 
        """

        return GMF_SWB.forward(self.data_nrcs, v, phi, theta, inverse_wave_age)


    def nrcs_inverse(self, nrcs, phi, theta, inverse_wave_age=1):
        """
        Estimates wind speed from NRCS
        
        Input
        -----
        nrcs : array like
            normalized radar cross section in linear units
        phi : array like
            wind direction in degrees with respect to sensor (between 0-360), 0 & 360 is blowing away, 180 is blowing towards
        theta : 
            incidence angle in degrees 
        inverse_wave_age : int | float | array like
            wave age of system, default = 1

        Returns
        -------
        result : array_like
            estimated wind speed 
        """

        return GMF_SWB.inverse(self.data_nrcs, nrcs, phi, theta, inverse_wave_age)


    def Dop_forward(self, v, phi, theta, inverse_wave_age=1):
        """
        Estimates Doppler from wind speed
        
        Input
        -----
        v : array like
            wind speed (m/s)
        phi : array like
            wind direction in degrees with respect to sensor (between 0-360), 0 & 360 is blowing away, 180 is blowing towards 
        theta : 
            incidence angle in degrees 
        inverse_wave_age : int | float | array like
            inverse wave age of system, default = 1

        Returns
        -------
        result : array_like
            estimated Doppler (Hz)
        """

        return GMF_SWB.forward(self.data_Doppler, v, phi, theta, inverse_wave_age)


    def Dop_inverse(self, dop, phi, theta, inverse_wave_age=1, pol='VV'):
        """
        Estimates wind speed from Doppler

        Input
        -----
        dop : array like
            doppler (Hz)
        phi : array like
            wind direction in degrees with respect to sensor (between 0-360), 0 & 360 is blowing away, 180 is blowing towards
        theta : 
            incidence angle in degrees 
        inverse_wave_age : int | float | array like
            wave age of system, default = 1

        Returns
        -------
        result : array_like
            estimated wind speed 
        """

        return GMF_SWB.inverse(self.data_Doppler, dop, phi, theta, inverse_wave_age)
