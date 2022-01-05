#!/usr/bin/env python

"""
Author: Lori Garzio on 1/5/2021
Last modified: 1/5/2021
Plot 3D ECOA transects from the CODAP-NA dataset
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean as cmo
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console
plt.rcParams.update({'font.size': 14})
np.set_printoptions(suppress=True)


def main(save_dir):
    codap_file = '/Users/garzio/Documents/rucool/Saba/NOAA_SOE2021/data/CODAP_NA_v2021.nc'
    lon_bounds = [-77, -71, -71, -77]
    lat_bounds = [37.5, 37.5, 40.5, 40.5]
    bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GMRTv3_9_20211118topo-mab-highres.grd'
    extent = [-77, -71.5, 37.5, 41.2]
    depth_max = 250
    rotate_view = [25, -70]
    yz_tickpad = 8
    yz_labelpad = 14
    plt_vars = {'CTDTEMP_ITS90': {'cmap': cmo.cm.thermal, 'ttl': 'Temperature (\N{DEGREE SIGN}C)',
                                  'vmin': 8, 'vmax': 26},
                'recommended_Salinity_PSS78': {'cmap': cmo.cm.haline, 'ttl': 'Salinity',
                                               'vmin': 29, 'vmax': 35},
                'pH': {'cmap': cmo.cm.matter, 'ttl': 'pH',
                       'vmin': 7.7, 'vmax': 8.3},
                'TALK': {'cmap': cmo.cm.matter, 'ttl': 'Total Alkalinity',
                         'vmin': 2000, 'vmax': 2400},
                'Aragonite': {'cmap': cmo.cm.matter, 'ttl': 'Aragonite Saturation',
                              'vmin': 1, 'vmax': 4}}

    # get ECOA data from CODAP-NA
    ds = xr.open_dataset(codap_file)
    codap_vars = dict(Cruise_ID=np.array([]),
                      Month_UTC=np.array([]),
                      Year_UTC=np.array([]),
                      Profile_number=np.array([]),
                      Latitude=np.array([]),
                      Longitude=np.array([]),
                      Depth=np.array([]),
                      Depth_bottom=np.array([]),
                      CTDTEMP_ITS90=np.array([]),
                      recommended_Salinity_PSS78=np.array([]),
                      pH_TS_insitu_calculated=np.array([]),
                      pH_TS_insitu_measured=np.array([]),
                      Aragonite=np.array([]),
                      TALK=np.array([]))

    # make sure the data are within the defined extent and find just the ECOA data
    for i, lon in enumerate(ds.Longitude.values):
        if Polygon(list(zip(lon_bounds, lat_bounds))).contains(Point(lon, ds.Latitude.values[i])):
            cid = ds.Cruise_ID.values[:, i]
            cid = [x.decode('UTF-8') for x in cid]
            cid = ''.join(cid).strip()
            if np.logical_or(cid == 'ECOA1', cid == 'ECOA2'):
                for key in codap_vars.keys():
                    if key == 'Cruise_ID':
                        codap_vars[key] = np.append(codap_vars[key], cid)
                    else:
                        codap_vars[key] = np.append(codap_vars[key], ds[key].values[i])

    df = pd.DataFrame(codap_vars)
    df['pH'] = df['pH_TS_insitu_measured']

    # use calculated pH if measured isn't available
    for idx, row in df.iterrows():
        if row['pH'] == -999:
            df.loc[row.name, 'pH'] = row.pH_TS_insitu_calculated

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

        try:
            sct = ax.scatter(df.Longitude, df.Latitude, df.Depth, c=data,
                             s=15, cmap=info['cmap'], vmin=info['vmin'], vmax=info['vmax'], zorder=2)
        except KeyError:
            sct = ax.scatter(df.Longitude, df.Latitude, df.Depth, c=data,
                             s=15, cmap=info['cmap'], zorder=2)

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
