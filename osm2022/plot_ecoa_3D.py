#!/usr/bin/env python

"""
Author: Lori Garzio on 1/5/2021
Last modified: 1/12/2021
Plot 3D ECOA transects from the CODAP-NA dataset
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import functions.common as cf
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console
plt.rcParams.update({'font.size': 14})
np.set_printoptions(suppress=True)


def main(save_dir):
    lon_bounds = [-77, -71, -71, -77]
    lat_bounds = [37.5, 37.5, 40.5, 40.5]
    bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GMRTv3_9_20211118topo-mab-highres.grd'
    extent = [-77, -71.5, 37.5, 41.2]
    depth_max = 250
    rotate_view = [20, -80]
    yz_tickpad = 8
    yz_labelpad = 14
    plt_vars = cf.plot_vars_ecoa()

    # get ECOA data from CODAP-NA
    df = cf.extract_ecoa_data(lon_bounds, lat_bounds)

    df = df[df.Depth <= depth_max]

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

        data = df[pv]
        data = data.replace({-999: np.nan})

        # white background to scatter points
        #sct = ax.scatter(df.Longitude, df.Latitude, df.Depth, c='k', s=18, zorder=2)

        try:
            sct = ax.scatter(df.Longitude, df.Latitude, df.Depth, c=data,
                             s=15, cmap=info['cmap'], vmin=info['vmin'], vmax=info['vmax'], zorder=3)
        except KeyError:
            sct = ax.scatter(df.Longitude, df.Latitude, df.Depth, c=data,
                             s=15, cmap=info['cmap'], zorder=3)

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

        sfile = os.path.join(save_dir, 'ECOA', f'{pv}_ECOA_3D.png')
        plt.savefig(sfile, dpi=200)
        plt.close()


if __name__ == '__main__':
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots/3D'
    main(save_directory)
