#!/usr/bin/env python

"""
Author: Lori Garzio on 1/14/2022
Last modified: 1/14/2022
Boxplots of surface- and bottom-averaged glider data for nearshore, midshelf, and offshore.
The box limits extend from the lower to upper quartiles, with a line at the median and a diamond symbol at the mean.
The whiskers extend from the box by 1.5x the inter-quartile range (IQR). Circles indicate outliers. Notch indicates
95% CI around the median.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import copy
import matplotlib.pyplot as plt
import functions.common as cf
plt.rcParams.update({'font.size': 12})
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console


def set_box_colors(bp, colors):
    for key in ['boxes', 'medians', 'fliers', 'means']:
        for patch, color in zip(bp[key], colors):
            patch.set_color(color)
            if key == 'boxes':
                patch.set_facecolor('none')
            elif key == 'means':
                patch.set_markerfacecolor(color)
                patch.set_markeredgecolor(color)
            elif key == 'fliers':
                patch.set_markeredgecolor(color)

    wc_colors = [x for pair in zip(colors, colors) for x in pair]
    for key in ['whiskers', 'caps']:
        for patch, color in zip(bp[key], wc_colors):
            patch.set_color(color)


def main(fname, save_dir, trsct):
    ds = xr.open_dataset(fname)
    deploy = '-'.join(fname.split('/')[-1].split('-')[0:2])

    if trsct:
        # select the transect
        time_range = cf.glider_time_selection()[deploy][trsct]
        ds = ds.sel(time=slice(time_range[0], time_range[1]))

    if deploy == 'ru30-20190717T1812':
        ds = ds.rename({'oxygen_mgl': 'oxygen_concentration_mgL_shifted', 'pH': 'ph_total_shifted',
                        'aragonite_saturation_state': 'saturation_aragonite'})

    # convert high oxygen values to nan
    oxyvars = [x for x in ds.data_vars if 'oxygen' in x]
    for ov in oxyvars:
        ds[ov][ds[ov] > 1000] = np.nan

    arr = np.array([], dtype='float32')
    divisions = dict(Nearshore=dict(Surface=arr, Bottom=arr),
                     Midshelf=dict(Surface=arr, Bottom=arr),
                     Offshore=dict(Surface=arr, Bottom=arr))

    # calculate MLD as depth in meters
    try:
        mld_meters = gsw.z_from_p(-ds.mld.values, ds.latitude.values)
    except AttributeError:
        mld_meters = gsw.z_from_p(-ds.mld.values, ds.profile_lat.values)
    attrs = {
        'actual_range': np.array([np.nanmin(mld_meters), np.nanmax(mld_meters)]),
        'observation_type': 'calculated',
        'units': 'm',
        'comment': 'Mixed Layer Depth calculated as the depth of max Brunt‐Vaisala frequency squared (N**2) from '
                   'Carvalho et al 2016. Calculated from MLD in dbar and latitude using gsw.z_from_p',
        'long_name': 'Mixed Layer Depth'
    }
    da = xr.DataArray(mld_meters, coords=ds.mld.coords, dims=ds.mld.dims,
                      name='mld_meters', attrs=attrs)
    ds['mld_meters'] = da

    ds2 = ds.swap_dims({'time': 'profile_time'})

    # convert to dataframe and group by profile
    df = ds2.to_dataframe()
    grouped = df.groupby('profile_time')

    # for each variable figure out if any of the variables were time-shifted and initialize an empty data dictionary
    plt_vars = cf.plot_vars_glider()
    drop_vars = []
    for pv, info in plt_vars.items():
        if '_shifted' in pv:  # if the data weren't time shifted, take the non-shifted variable
            if np.sum(~np.isnan(ds[pv].values)) == 0:
                drop_vars.append(pv)
            else:
                plt_vars[pv]['data'] = copy.deepcopy(divisions)
        else:
            plt_vars[pv]['data'] = copy.deepcopy(divisions)
    for dv in drop_vars:
        new_dv = dv.split('_shifted')[0]
        plt_vars[new_dv] = plt_vars[dv]
        plt_vars[new_dv]['data'] = copy.deepcopy(divisions)
        plt_vars.pop(dv)

    # create empty dictionary for MLD
    mld_dict = dict(ttl='Mixed Layer Depth (m)',
                    data=dict(Nearshore=arr, Midshelf=arr, Offshore=arr))

    # for each profile, split the data into correct divisions and append data
    for group in grouped:
        temp_df = cf.depth_bin(group[1])
        temp_df.dropna(subset=['depth'], inplace=True)
        profile_mld = np.nanmean(temp_df.mld_meters)
        if np.isnan(profile_mld):  # if MLD isn't defined, throw out the profile
            continue

        # define the location on the shelf
        profile_maxz = np.nanmax(temp_df.depth)
        if profile_maxz < 36:
            shelf_loc = 'Nearshore'
        elif np.logical_and(profile_maxz >= 36, profile_maxz <= 100):
            shelf_loc = 'Midshelf'
        else:
            shelf_loc = 'Offshore'

        # append MLD data
        mld_dict['data'][shelf_loc] =np.append(mld_dict['data'][shelf_loc], profile_mld)

        # separate surface and bottom waters
        surface_df = temp_df[temp_df.depth < profile_mld]
        bottom_df = temp_df[temp_df.depth > profile_mld]

        # append 1m depth-binned variable data
        for pv, info in plt_vars.items():
            dict_append = info['data'][shelf_loc]
            dict_append['Surface'] = np.append(dict_append['Surface'], surface_df[pv].values)
            dict_append['Bottom'] = np.append(dict_append['Bottom'], bottom_df[pv].values)

    # boxplot of MLD
    bplot = []
    labels = []
    for k, v in mld_dict['data'].items():
        bplot.append(list(v))
        labels.append(f'{k}\nn={len(v)}')

    fig, ax = plt.subplots(figsize=(8, 10))

    # customize the boxplot elements
    medianprops = dict(color='black')
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black')
    boxprops = dict(facecolor='darkgray')

    box = ax.boxplot(bplot, patch_artist=True, labels=labels, showmeans=True, notch=True,
                     medianprops=medianprops, meanprops=meanpointprops, boxprops=boxprops)
    ax.set_ylabel('Mixed Layer Depth (m)')

    ax.invert_yaxis()

    sfilename = f'{deploy}_boxplot_mld.png'
    sfile = os.path.join(save_dir, 'boxplot', sfilename)
    plt.savefig(sfile, dpi=300)
    plt.close()

    box_colors = ['tab:blue', 'tab:orange', 'k']

    # iterate through each variable and plot boxplots
    for pv, info in plt_vars.items():
        surface_data = [list(info['data']['Nearshore']['Surface']),
                        list(info['data']['Midshelf']['Surface']),
                        list(info['data']['Offshore']['Surface'])]
        bottom_data = [list(info['data']['Nearshore']['Bottom']),
                       list(info['data']['Midshelf']['Bottom']),
                       list(info['data']['Offshore']['Bottom'])]

        fig, ax = plt.subplots(figsize=(8, 10))

        # customize the boxplot elements
        meanpointprops = dict(marker='D')

        bp_surf = ax.boxplot(surface_data, positions=[1, 2, 3], widths=0.6, patch_artist=True, showmeans=True,
                             notch=True, meanprops=meanpointprops)
        bp_bottom = ax.boxplot(bottom_data, positions=[5, 6, 7], widths=0.6, patch_artist=True, showmeans=True,
                               notch=True, meanprops=meanpointprops)

        # set box colors
        set_box_colors(bp_surf, box_colors)
        set_box_colors(bp_bottom, box_colors)

        # draw temporary lines and use them to create a legend
        plt.plot([], c='tab:blue', label='Nearshore')
        plt.plot([], c='tab:orange', label='Midshelf')
        plt.plot([], c='k', label='Offshore')
        plt.legend(fontsize=10)

        # set axes labels
        ax.set_xticks([2, 6])
        ax.set_xticklabels(['Surface', 'Bottom'])
        ax.set_ylabel(info['ttl'])

        # draw a horizontal line between sections
        ylims = ax.get_ylim()
        ax.vlines(4, ylims[0], ylims[1], colors='k')
        ax.set_ylim(ylims)

        sfilename = f'{deploy}_boxplot_{pv}.png'
        sfile = os.path.join(save_dir, 'boxplot', sfilename)
        plt.savefig(sfile, dpi=300)
        plt.close()


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys_mld.nc'
    # ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/sbu01-20210720T1628-profile-sci-delayed-qc_shifted_co2sys_mld.nc'
    # ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20190717T1812-delayed_mld.nc'
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots'
    transect = 'first_transect'  # first_transect, last_transect (for sbu) False
    main(ncfile, save_directory, transect)
