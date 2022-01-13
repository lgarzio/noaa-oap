#!/usr/bin/env python

"""
Author: Lori Garzio on 1/12/2022
Last modified: 1/12/2022
"""

import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console


def main(save_dir, transect):
    bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
    bathy = xr.open_dataset(bathymetry)
    if transect == 'all':
        lon_bounds = [-77, -71, -71, -77]
        lat_bounds = [37.5, 37.5, 40.5, 40.5]
    elif transect == 'NJ':
        lon_bounds = [-74, -72, -72, -74]
        lat_bounds = [40, 38.8, 40.5, 40.5]
    elif transect == 'DE':
        lon_bounds = [-76, -73, -73, -75]
        lat_bounds = [39, 37.5, 37.9, 39]

    # get ECOA data from CODAP-NA
    df = cf.extract_ecoa_data(lon_bounds, lat_bounds)

    df = df[df.Depth <= 250]

    sfile = os.path.join(save_dir, 'ecoa_tracks', f'ECOA_track_{transect}.png')

    fig, ax = plt.subplots(figsize=(11, 8), subplot_kw=dict(projection=ccrs.Mercator()))
    plt.subplots_adjust(right=0.82)

    title = f'ECOA track: {transect}\nsampling depth <= 250m'

    extent = [-76, -71.5, 37.4, 40.6]

    kwargs = dict()
    kwargs['bathy'] = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                                lat=slice(extent[2] - .1, extent[3] + .1))

    kwargs['title'] = title
    kwargs['landcolor'] = 'none'

    fig, ax = pf.surface_track_df(fig, ax, df, extent, **kwargs)

    plt.savefig(sfile, dpi=200)
    plt.close()


if __name__ == '__main__':
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots'
    transect = 'DE'  # 'NJ' 'DE' 'all'
    main(save_directory, transect)
