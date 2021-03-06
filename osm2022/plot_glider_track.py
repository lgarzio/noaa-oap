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


def main(save_dir, ff, mdates, trsct):
    bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
    bathy = xr.open_dataset(bathymetry)
    for fname in ff:
        ds = xr.open_dataset(fname)
        deploy = '-'.join(fname.split('/')[-1].split('-')[0:2])
        sfile = os.path.join(save_dir, 'glider_tracks', f'{deploy}_track.png')

        if mdates:
            md = cf.osm_mask_dates()
            deploy_md = md[deploy]
            if deploy_md:
                sfile = os.path.join(save_dir, 'glider_tracks', f'{deploy}_track_masked.png')
                for md in deploy_md:
                    md_idx = np.where(np.logical_and(pd.to_datetime(ds.time.values) >= md[0], pd.to_datetime(ds.time.values) <= md[1]))[0]
                    try:
                        ds.latitude[md_idx] = np.nan
                        ds.longitude[md_idx] = np.nan
                    except AttributeError:
                        ds.profile_lat[md_idx] = np.nan
                        ds.profile_lon[md_idx] = np.nan

        if trsct:
            try:
                time_range = cf.glider_time_selection()[deploy][trsct]
                ds = ds.sel(time=slice(time_range[0], time_range[1]))
                sfile = os.path.join(save_dir, 'glider_tracks', f'{deploy}_track_{trsct}.png')
            except KeyError:
                print(f'{trsct} for {deploy} not specified')
                continue

        fig, ax = plt.subplots(figsize=(11, 8), subplot_kw=dict(projection=ccrs.Mercator()))
        plt.subplots_adjust(right=0.82)

        t0 = pd.to_datetime(np.nanmin(ds.time)).strftime('%Y-%m-%dT%H:%M')
        tf = pd.to_datetime(np.nanmax(ds.time)).strftime('%Y-%m-%dT%H:%M')

        title = f'{deploy.split("-")[0]} track: {t0} to {tf}'

        extent = [-75, -71.5, 37.8, 41.2]

        kwargs = dict()
        kwargs['timevariable'] = 'time'
        kwargs['bathy'] = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                                    lat=slice(extent[2] - .1, extent[3] + .1))

        kwargs['title'] = title
        kwargs['landcolor'] = 'none'

        fig, ax = pf.glider_track(fig, ax, ds, extent, **kwargs)

        plt.savefig(sfile, dpi=200)
        plt.close()


if __name__ == '__main__':
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots'
    files = ['/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20190717T1812-delayed_mld.nc',
             '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys_mld.nc',
             '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/sbu01-20210720T1628-profile-sci-delayed-qc_shifted_co2sys_mld.nc']
    mask_dates = False  # True False
    transect = 'last_transect'  # None first_transect last_transect
    main(save_directory, files, mask_dates, transect)
