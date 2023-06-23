#!/usr/bin/env python

"""
Author: Lori Garzio on 1/14/2022
Last modified: 6/22/2023
Plot cross-shelf transect for a glider deployment (1-m depth binned 1-km distance binned contour plot)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def main(fname, save_dir, plt_mld, ymax, xlims, ac):
    ds = xr.open_dataset(fname)
    deployment = ds.title
    save_dir = os.path.join(save_dir, 'transect_contourf', deployment)
    os.makedirs(save_dir, exist_ok=True)

    # get plotting variables and initialize empty list to append data
    plt_vars = cf.plot_vars_glider()
    for pv, info in plt_vars.items():
        plt_vars[pv]['data'] = []

    depth_max = np.ceil(np.nanmax(ds.depth.values))
    try:
        depth_var = ds.depth_interpolated.name
    except AttributeError:
        depth_var = ds.depth.name

    # convert to dataframe and group by profile
    df = ds.to_dataframe()
    grouped = df.groupby('time')

    # initialize an empty array for distance from shore and MLD for each profile for plotting
    profile_dist_from_shore = np.array([])
    profile_mld_meters = np.array([])

    for group in grouped:
        kwargs = {
            'depth_max': depth_max,
            'depth_var': depth_var
        }
        temp_df = cf.depth_bin(group[1], **kwargs)
        profile_dist_from_shore = np.append(profile_dist_from_shore, np.nanmean(temp_df['dist_from_shore_km']))
        profile_mld_meters = np.append(profile_mld_meters, np.nanmean(temp_df['mld']))

        # add data to each variable for each profile
        for pv, info in plt_vars.items():
            try:
                info['data'].append(temp_df[pv].values.tolist())
            except KeyError:
                continue

    if plt_mld:
        df_mld = pd.DataFrame({'profile_mld_meters': profile_mld_meters})

        # group by distance from shore rounded to the nearest 1 km
        df_mld['distance_from_shore'] = np.round(profile_dist_from_shore)
        df_mld = df_mld.groupby('distance_from_shore').mean()

    # add the cold pool contour to each plot
    if ac:
        temperature_df = pd.DataFrame(plt_vars['temperature']['data'])

        # group by distance from shore rounded to the nearest 1 km
        temperature_df['distance_from_shore'] = np.round(profile_dist_from_shore)
        temperature_df = temperature_df.groupby('distance_from_shore').mean()
        temperature_df = temperature_df.transpose()
        xtemperature, ytemperature = np.meshgrid(temperature_df.columns.values, temperature_df.index.values)

        # find the cold pool 10C isotherm intersection with the bottom
        cp = cf.glider_coldpool_extent_updated()[deployment]['transect']
        cpdf = temperature_df[cp].dropna()
        cp_depth = np.nanmax(cpdf.index)

    # iterate through each variable, turn the data into a binned dataframe and generate a contourf plot
    for pv, info in plt_vars.items():
        extend = 'both'
        if 'chlorophyll' in pv:
            extend = 'max'

        pv_df = pd.DataFrame(info['data'])
        if len(pv_df) == 0:
            continue

        # remove bad DO values
        if 'oxygen' in pv:
            pv_df[pv_df > 1000] = np.nan

        # group by distance from shore rounded to the nearest 1 km
        pv_df['distance_from_shore'] = np.round(profile_dist_from_shore)
        pv_df = pv_df.groupby('distance_from_shore').mean()

        # rows = depth, columns = distance from shore
        pv_df = pv_df.transpose()

        x, y = np.meshgrid(pv_df.columns.values, pv_df.index.values)

        fig, ax = plt.subplots(figsize=(11, 8))
        plt.subplots_adjust(left=0.08, right=1)

        cs = plt.contourf(x, y, pv_df, levels=info['levels'], cmap=info['cmap'], extend=extend)
        plt.colorbar(cs, ax=ax, label=info['ttl'], pad=0.02)

        if ymax:
            ax.set_ylim([0, ymax])

        if xlims:
            ax.set_xlim(xlims)

        if ac:
            # plot a contour of the coldpool
            ax.contour(xtemperature, ytemperature, temperature_df, [10], colors='white', linewidths=1)

            # plot a horizontal line where the 10C isobath intersects with the bottom
            plot_xlims = ax.get_xlim()
            ax.hlines(cp_depth, plot_xlims[0], cp, colors='k')

        ax.invert_yaxis()

        if plt_mld:
            ax.plot(df_mld.index.values, df_mld.profile_mld_meters.values, ls='-', lw=1.5, color='k')

        plt.setp(ax, ylabel='Depth (m)', xlabel='Distance from Shore (km)')

        sfile = os.path.join(save_dir, f'{deployment}_{pv}_transect.png')
        plt.savefig(sfile, dpi=200)
        plt.close()


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/NOAA_OAP/gliderdata_dac/ru30-20210716T1804-delayed-transect.nc'
    save_directory = '/Users/garzio/Documents/rucool/Saba/NOAA_OAP/wcr_analysis'
    plot_mld = True  # True False
    ymaximum = False  # 180  # max depth for y-axis  False
    x_limits = False  # [10, 200]  # optional limits for x-axis  False
    add_coldpool = True
    main(ncfile, save_directory, plot_mld, ymaximum, x_limits, add_coldpool)
