#!/usr/bin/env python

"""
Author: Lori Garzio on 6/13/2023
Last modified: 6/13/2023
"""

import numpy as np
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
plt.rcParams.update({'font.size': 13})


def main(fname, save_dir):
    bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
    save_dir = os.path.join(save_dir, 'glider_tracks')
    os.makedirs(save_dir, exist_ok=True)

    ds = xr.open_dataset(fname)
    ds = ds.swap_dims({'row': 'time'})
    deployment = ds.title

    glider_region = cf.glider_region(ds)  # define the glider region

    fig, ax = plt.subplots(figsize=(11, 8), subplot_kw=dict(projection=ccrs.Mercator()))
    plt.subplots_adjust(right=0.82)

    t0 = pd.to_datetime(np.nanmin(ds.time)).strftime('%Y-%m-%dT%H:%M')
    tf = pd.to_datetime(np.nanmax(ds.time)).strftime('%Y-%m-%dT%H:%M')

    title = f'{deployment}\n{t0} to {tf}'
    extent = glider_region['extent']
    bathy = xr.open_dataset(bathymetry)

    kwargs = dict()
    kwargs['bathy'] = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                                lat=slice(extent[2] - .1, extent[3] + .1))

    kwargs['title'] = title
    kwargs['landcolor'] = 'none'
    pf.glider_track(fig, ax, ds, extent, **kwargs)

    sname = os.path.join(save_dir, '{}_glider-track.png'.format(deployment))
    plt.savefig(sname, dpi=200)
    plt.close()


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/NOAA_OAP/gliderdata_dac/ru33-20180801T1323.nc'
    savedir = '/Users/garzio/Documents/rucool/Saba/NOAA_OAP/wcr_analysis'
    main(ncfile, savedir)
