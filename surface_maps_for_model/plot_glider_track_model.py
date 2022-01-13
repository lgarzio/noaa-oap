#!/usr/bin/env python

"""
Author: Lori Garzio on 1/11/2022
Last modified: 1/11/2022
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


def main(save_dir, ff, hdates):
    bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
    bathy = xr.open_dataset(bathymetry)
    colors = ['yellow', 'magenta']
    for f in ff:
        ds = xr.open_dataset(f)
        try:
            ds = ds.swap_dims({'row': 'time'})
        except ValueError:
            print('time is already a dimension')

        df = ds.to_dataframe()

        deploy = '-'.join(f.split('/')[-1].split('-')[0:2])

        sfile = os.path.join(save_dir, f'{deploy}_track.png')

        fig, ax = plt.subplots(figsize=(11, 8), subplot_kw=dict(projection=ccrs.Mercator()))
        plt.subplots_adjust(right=0.82)

        t0 = pd.to_datetime(np.nanmin(ds['time'])).strftime('%Y-%m-%dT%H:%M')
        tf = pd.to_datetime(np.nanmax(ds['time'])).strftime('%Y-%m-%dT%H:%M')

        title = f'{deploy.split("-")[0]} track: {t0} to {tf}'

        extent = [-75, -71.5, 37.8, 41.2]

        kwargs = dict()
        kwargs['timevariable'] = 'time'
        kwargs['bathy'] = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                                    lat=slice(extent[2] - .1, extent[3] + .1))

        kwargs['title'] = title
        kwargs['landcolor'] = 'none'

        fig, ax = pf.glider_track(fig, ax, ds, extent, **kwargs)

        # plot vertical lines
        if hdates:
            hd = cf.model_highlight_dates()
            deploy_hd = hd[deploy]
            if deploy_hd:
                for idx, dhd in enumerate(deploy_hd):
                    df_dhd = df[np.logical_and(df.index > dhd[0], df.index < dhd[1])]
                    ax.scatter(df_dhd.longitude, df_dhd.latitude, color='k', marker='.', s=125,
                               transform=ccrs.PlateCarree(), zorder=15)
                    ax.scatter(df_dhd.longitude, df_dhd.latitude, color=colors[idx], marker='.', s=60,
                               transform=ccrs.PlateCarree(), zorder=16)

        plt.savefig(sfile, dpi=200)
        plt.close()


if __name__ == '__main__':
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/for_model/plots'
    files = ['/Users/garzio/Documents/rucool/Saba/gliderdata/2021/sbu01-20211102T1528/sbu01-20211102T1528-profile-sci-rt.nc',
             '/Users/garzio/Documents/rucool/Saba/gliderdata/2021/ru30-20210716T1804/delayed/testserver_qc/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys.nc',
             '/Users/garzio/Documents/rucool/Saba/gliderdata/2021/ru30-20210226T1647/delayed/ru30-20210226T1647-profile-sci-delayed_shifted_qc.nc']
    highlight_dates = True  # True False
    main(save_directory, files, highlight_dates)
