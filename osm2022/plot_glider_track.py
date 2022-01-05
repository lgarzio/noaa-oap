#!/usr/bin/env python

"""
Author: Lori Garzio on 1/3/2022
Last modified: 1/3/2022
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})


def main(save_dir, ff, mdates):
    bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
    bathy = xr.open_dataset(bathymetry)
    for f in ff:
        ds = xr.open_dataset(f)
        filename = f.split('/')[-1]
        if filename == 'ru30_summer_19.nc':
            deploy = 'ru30-summer2019'
            ds = ds.swap_dims({'time': 'UnixTime'})
            ds = ds.rename({'Latitude': 'latitude', 'Longitude': 'longitude'})
            tvar = 'UnixTime'
        elif 'ru30' in filename:
            deploy = 'ru30-summer2021'
            tvar = 'time'
        elif 'sbu01' in filename:
            deploy = 'sbu01-summer2021'
            tvar = 'time'

        sfile = os.path.join(save_dir, 'tracks', f'{deploy}_track.png')

        if mdates:
            md = cf.mask_dates()
            deploy_md = md[deploy]
            if deploy_md:
                sfile = os.path.join(save_dir, 'tracks', f'{deploy}_track_masked.png')
                for md in deploy_md:
                    md_idx = np.where(np.logical_and(pd.to_datetime(ds[tvar].values) >= md[0], pd.to_datetime(ds[tvar].values) <= md[1]))[0]
                    ds.latitude[md_idx] = np.nan
                    ds.longitude[md_idx] = np.nan

        fig, ax = plt.subplots(figsize=(11, 8), subplot_kw=dict(projection=ccrs.Mercator()))
        plt.subplots_adjust(right=0.82)

        t0 = pd.to_datetime(np.nanmin(ds[tvar])).strftime('%Y-%m-%dT%H:%M')
        tf = pd.to_datetime(np.nanmax(ds[tvar])).strftime('%Y-%m-%dT%H:%M')

        title = f'{deploy.split("-")[0]} track: {t0} to {tf}'

        extent = [-75, -71.5, 37.8, 41.2]

        kwargs = dict()
        kwargs['timevariable'] = tvar
        kwargs['bathy'] = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                                    lat=slice(extent[2] - .1, extent[3] + .1))

        kwargs['title'] = title
        kwargs['landcolor'] = 'none'

        fig, ax = pf.glider_track(fig, ax, ds, extent, **kwargs)

        plt.savefig(sfile, dpi=200)
        plt.close()


if __name__ == '__main__':
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots'
    files = ['/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30_summer_19.nc',
             '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/sbu01-20210720T1628-profile-sci-delayed-qc_shifted_co2sys.nc',
             '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys.nc']
    mask_dates = True  # True False
    main(save_directory, files, mask_dates)
