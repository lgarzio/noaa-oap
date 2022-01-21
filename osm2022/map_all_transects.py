#!/usr/bin/env python

"""
Author: Lori Garzio on 1/19/2022
Last modified: 1/19/2022
Plot a map of glider and ECOA transects
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


def main(save_dir, file_list, trsct):
    # set up map
    fig, ax = plt.subplots(figsize=(11, 8), subplot_kw=dict(projection=ccrs.Mercator()))
    extent = [-76, -71.3, 37.5, 41]
    margs = dict()
    margs['landcolor'] = 'none'
    pf.add_map_features(ax, extent, **margs)
    bathymetry = '/Users/garzio/Documents/rucool/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
    bathy = xr.open_dataset(bathymetry)
    bathy = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                      lat=slice(extent[2] - .1, extent[3] + .1))
    pf.add_bathymetry(ax, bathy)

    # plot ECOA data
    for ecoa_trsct in ['NJ', 'DE']:
        lon_bounds, lat_bounds = cf.ecoa_transect_bounds(ecoa_trsct)

        # get ECOA data from CODAP-NA
        df = cf.extract_ecoa_data(lon_bounds, lat_bounds)
        df = df[df.Depth <= 250]

        years = [2015, 2018]
        colors = ['#225ea8', 'cyan']
        labels = ['ECOA-2015', 'ECOA-2018']

        for i, yr in enumerate(years):
            df_yr = df[df['Year_UTC'] == yr]
            ax.scatter(df_yr.Longitude.values, df_yr.Latitude.values, color='k', marker='.', s=130,
                       transform=ccrs.PlateCarree(), zorder=10)
            ax.scatter(df_yr.Longitude.values, df_yr.Latitude.values, color=colors[i], marker='.', s=60,
                       transform=ccrs.PlateCarree(), zorder=11, label=labels[i])

    for fname in file_list:
        ds = xr.open_dataset(fname)
        deploy = '-'.join(fname.split('/')[-1].split('-')[0:2])
        if deploy == 'sbu01-20210720T1628':
            color = '#fed976'  # yellow
            label = 'sbu01-summer21'
        elif deploy == 'ru30-20190717T1812':
            color = '#bd0026'  # red
            label = 'ru30-summer19'
        elif deploy == 'ru30-20210716T1804':
            color = '#fd8d3c'  # orange
            label = 'ru30-summer21'

        # select the transect
        time_range = cf.glider_time_selection()[deploy][trsct]
        ds = ds.sel(time=slice(time_range[0], time_range[1]))

        lon = ds.profile_lon.values
        lat = ds.profile_lat.values

        ax.scatter(lon, lat, color='k', marker='.', s=50, transform=ccrs.PlateCarree(), zorder=10)
        ax.scatter(lon, lat, color=color, marker='.', s=10, transform=ccrs.PlateCarree(), zorder=11, label=label)

    handles, labels = plt.gca().get_legend_handles_labels()  # only show one set of legend labels
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), framealpha=.8)

    # set all legend points to be the same size and have an edgecolor = black
    for ha in ax.legend_.legendHandles:
        ha.set_edgecolor('k')
        ha._sizes = [150]

    sfile = os.path.join(save_dir, 'transects', 'glider_ecoa_transects.png')
    plt.savefig(sfile, dpi=200)
    plt.close()


if __name__ == '__main__':
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots'
    files = ['/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20190717T1812-delayed_mld.nc',
             '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys_mld.nc',
             '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/sbu01-20210720T1628-profile-sci-delayed-qc_shifted_co2sys_mld.nc']
    glider_transect = 'first_transect'  # None first_transect
    main(save_directory, files, glider_transect)
