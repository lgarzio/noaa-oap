#!/usr/bin/env python

"""
Author: Lori Garzio on 1/3/2022
Last modified: 1/5/2022
Plot 3D maps of glider data.
"""

import datetime as dt
import os
import xarray as xr
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # need this for 3D scatter plot
import matplotlib.pyplot as plt
import cmocean as cmo
import functions.common as cf
plt.rcParams.update({'font.size': 14})


def main(save_dir, ff, mdates):
    bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GMRTv3_9_20211118topo-mab-highres.grd'
    if mdates:
        depth_max = 100
    else:
        depth_max = 250
    rotate_view = [25, -70]
    extent = [-75, -71.5, 37.8, 41.2]
    yz_tickpad = 8
    yz_labelpad = 14
    plt_vars = {'oxygen_concentration_mgL_shifted': {'cmap': cmo.cm.oxy, 'ttl': 'Oxygen (mg/L)',
                                                     'vmin': 5, 'vmax': 11},
                'temperature': {'cmap': cmo.cm.thermal, 'ttl': 'Temperature (\N{DEGREE SIGN}C)',
                                'vmin': 8, 'vmax': 26},
                'salinity': {'cmap': cmo.cm.haline, 'ttl': 'Salinity',
                             'vmin': 29, 'vmax': 35},
                'chlorophyll_a': {'cmap': cmo.cm.algae, 'ttl': 'Chlorophyll ({}g/L)'.format(chr(956)),
                                  'vmin': 0, 'vmax': 8},
                'ph_total_shifted': {'cmap': cmo.cm.matter, 'ttl': 'pH',
                                     'vmin': 7.7, 'vmax': 8.3},
                'total_alkalinity': {'cmap': cmo.cm.matter, 'ttl': 'Total Alkalinity',
                                     'vmin': 2000, 'vmax': 2400},
                'saturation_aragonite': {'cmap': cmo.cm.matter, 'ttl': 'Aragonite Saturation',
                                         'vmin': 1, 'vmax': 4}}

    if len(ff) == 1:  # 2019 deployment
        savename = 'ru30-summer2019'
    else:  # ru30 and sbu01 on the same map
        savename = 'ru30_sbu01-summer2021'

    # manipulate bathymetry
    bathy = xr.open_dataset(bathymetry)
    bathy = bathy.sel(lon=slice(extent[0], extent[1]),
                      lat=slice(extent[2], extent[3]))
    x, y = np.meshgrid(bathy.lon.values, bathy.lat.values)
    alt = bathy.altitude
    lm_idx = np.logical_and(alt > 0, alt > 0)
    deep_idx = np.logical_and(alt < -depth_max, alt < -depth_max)

    # mask land elevation and deep values
    alt.values[lm_idx] = np.nan
    alt.values[deep_idx] = np.nan

    # find the coastline
    coast = np.empty(shape=np.shape(alt))
    coast[:] = np.nan

    # create a land variable
    landmask = alt.values.copy()
    landmask[lm_idx] = 0
    landmask[~lm_idx] = np.nan

    for pv, info in plt_vars.items():
        # create 3D plot
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": "3d",
                                                             "computed_zorder": False})

        # change the scaling of the plot area
        ax.set_box_aspect(aspect=(2, 2, 1))

        # Remove gray panes and axis grid
        ax.xaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.fill = False
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.fill = False
        ax.zaxis.pane.set_edgecolor('white')
        ax.grid(False)

        ax.plot_surface(x, y, -alt, color='lightgray', zorder=-1, alpha=.3)  # plot bathymetry
        ax.plot_wireframe(x, y, -alt, color='gray', zorder=0, alpha=.3)  # plot wireframe bathymetry
        ax.plot_surface(x, y, landmask, color='tan', zorder=1, shade=False)  # plot land

        for f in ff:
            ds = xr.open_dataset(f)
            filename = f.split('/')[-1]
            if filename == 'ru30_summer_19.nc':
                deploy = 'ru30-summer2019'
                ds = ds.swap_dims({'time': 'UnixTime'})
                tvar = 'UnixTime'
                ds = ds.rename({'Pressure': 'pressure', 'Temperature': 'temperature', 'Salinity': 'salinity',
                                'Chlorophyll': 'chlorophyll_a', 'Oxygen_mgL': 'oxygen_concentration_mgL_shifted',
                                'pH': 'ph_total_shifted', 'TotalAlkalinity': 'total_alkalinity',
                                'AragoniteSaturationState': 'saturation_aragonite', 'Latitude': 'latitude',
                                'Longitude': 'longitude', 'Conductivity': 'conductivity',
                                'Oxygen_molar': 'oxygen_concentration_shifted',
                                'pHReferenceVoltage': 'sbe41n_ph_ref_voltage'})

                ds['oxygen_concentration_mgL_shifted'][ds['oxygen_concentration_mgL_shifted'] > 1000] = np.nan
            elif 'ru30' in filename:
                deploy = 'ru30-summer2021'
                tvar = 'time'
            elif 'sbu01' in filename:
                deploy = 'sbu01-summer2021'
                tvar = 'time'

            data = ds[pv]

            if '_shifted' in pv:  # if the data weren't time shifted, take the non-shifted variable
                if np.sum(~np.isnan(data.values)) == 0:
                    new_pv = pv.split('_shifted')[0]
                    ds[new_pv][ds[new_pv] > 1000] = np.nan
                    data = ds[new_pv]

            sfile = os.path.join(save_dir, 'full_track', f'{pv}_{savename}_3D.png')

            if mdates:
                md = cf.mask_dates()
                deploy_md = md[deploy]
                if deploy_md:
                    sfile = os.path.join(save_dir, 'cross_shelf_only', f'{pv}_{savename}_3D_masked.png')
                    for md in deploy_md:
                        md_idx = np.where(np.logical_and(pd.to_datetime(ds[tvar].values) >= md[0],
                                                         pd.to_datetime(ds[tvar].values) <= md[1]))[0]
                        data[md_idx] = np.nan

            try:
                depth = ds.depth.values
            except AttributeError:
                depth = ds.pressure.values

            try:
                sct = ax.scatter(ds.longitude.values, ds.latitude.values, depth, c=data.values,
                                 s=2, cmap=info['cmap'], vmin=info['vmin'], vmax=info['vmax'], zorder=2)
            except KeyError:
                sct = ax.scatter(ds.longitude.values, ds.latitude.values, depth, c=data.values,
                                 s=2, cmap=info['cmap'], zorder=2)

        cbar = plt.colorbar(sct, label=info['ttl'], extend='both', shrink=0.7, pad=0.08)
        ax.invert_zaxis()
        ax.set_zlabel('Depth (m)', labelpad=yz_labelpad)
        ax.set_ylabel('Latitude', labelpad=yz_labelpad)
        ax.set_xlabel('Longitude', labelpad=14)

        # add space between ticks and tick labels
        ax.tick_params(axis='y', which='major', pad=yz_tickpad)
        ax.tick_params(axis='z', which='major', pad=yz_tickpad)

        if rotate_view:
            # ax.view_init(elev=30, azim=-60)  # defaults
            ax.view_init(elev=rotate_view[0], azim=rotate_view[1])  # rotate the view

        plt.savefig(sfile, dpi=200)
        plt.close()


if __name__ == '__main__':
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots/3D'
    # files = ['/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30_summer_19.nc']
    files = ['/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys.nc',
             '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/sbu01-20210720T1628-profile-sci-delayed-qc_shifted_co2sys.nc']
    mask_dates = True  # True False
    main(save_directory, files, mask_dates)
