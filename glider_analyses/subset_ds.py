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


def main(fname, trsct):
    ds = xr.open_dataset(fname)
    ds = ds.swap_dims({'row': 'time'})
    deployment = ds.title
    time_selection = cf.glider_time_selection_updated()[deployment][trsct]
    shore_coords = cf.glider_shore_locations_updated()[deployment][trsct]

    ds = ds.sel(time=slice(np.nanmin(time_selection), np.nanmax(time_selection)))

    # calculate distance from shore
    geod = Geodesic.WGS84
    dist_from_shore_km = np.array(np.empty(len(ds.latitude)))
    dist_from_shore_km[:] = np.nan
    profile_lats = np.unique(ds.latitude)
    for pl in profile_lats:
        if ~np.isnan(pl):  # if latitude is not nan
            pl_idx = np.where(ds.latitude.values == pl)[0]
            plon = ds.longitude.values[pl_idx]
            g = geod.Inverse(shore_coords[0], shore_coords[1], pl, np.unique(plon)[0])
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
    da = xr.DataArray(dist_from_shore_km, coords=ds.latitude.coords, dims=ds.latitude.dims,
                      name='dist_from_shore_km', attrs=attrs)
    ds['dist_from_shore_km'] = da

    savefile = f'{fname.split(".nc")[0]}-transect.nc'
    ds.to_netcdf(savefile)


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/NOAA_OAP/gliderdata_dac/ru33-20180801T1323.nc'
    transect = 'second_transect'  # first_transect, last_transect (for sbu)
    main(ncfile, transect)
