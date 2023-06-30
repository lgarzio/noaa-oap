#!/usr/bin/env python

"""
Author: Lori Garzio on 6/21/2023
Last modified: 6/21/2023
Generate subset .nc files with one cross-shelf transect, and calculate distance from shore.
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt
import functions.common as cf
plt.rcParams.update({'font.size': 14})
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console


def main(fname):
    ds = xr.open_dataset(fname)
    ds = ds.swap_dims({'row': 'time'})
    deployment = ds.title
    time_selections = cf.glider_time_selection_updated()[deployment]
    for trsct, ts in time_selections.items():
        shore_coords = cf.glider_shore_locations_updated()[deployment][trsct]

        ds_sel = ds.sel(time=slice(np.nanmin(ts), np.nanmax(ts)))

        # calculate distance from shore
        geod = Geodesic.WGS84
        dist_from_shore_km = np.array(np.empty(len(ds_sel.latitude)))
        dist_from_shore_km[:] = np.nan
        profile_lats = np.unique(ds_sel.latitude)
        for pl in profile_lats:
            if ~np.isnan(pl):  # if latitude is not nan
                pl_idx = np.where(ds_sel.latitude.values == pl)[0]
                plon = ds_sel.longitude.values[pl_idx]
                g = geod.Inverse(shore_coords[1], shore_coords[0], pl, np.unique(plon)[0])
                dist_km = g['s12'] * .001
                dist_from_shore_km[pl_idx] = dist_km

        # add distance from shore to dataset
        attrs = {
            'actual_range': np.array([np.nanmin(dist_from_shore_km), np.nanmax(dist_from_shore_km)]),
            'observation_type': 'calculated',
            'units': 'km',
            'comment': f'distance from shore, calculated from shore location {shore_coords}',
            'shore_point': shore_coords,
            'long_name': 'Distance from Shore'
        }
        da = xr.DataArray(dist_from_shore_km, coords=ds_sel.latitude.coords, dims=ds_sel.latitude.dims,
                          name='dist_from_shore_km', attrs=attrs)
        ds_sel['dist_from_shore_km'] = da

        savefile = f'{fname.split(".nc")[0]}-{trsct}.nc'
        ds_sel.to_netcdf(savefile)


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/NOAA_OAP/gliderdata_dac/SBU01-20220805T1855-delayed.nc'
    main(ncfile)
