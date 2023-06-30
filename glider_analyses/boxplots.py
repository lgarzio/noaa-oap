#!/usr/bin/env python

"""
Author: Lori Garzio on 1/14/2022
Last modified: 6/28/2023
Boxplots of surface- and bottom-averaged glider data for nearshore, midshelf, and offshore. Data are 1-m depth binned,
1-km distance binned. Shelf locations are defined as the profile maximum depth: nearshore = profile depth < 36m,
midshelf = profile depth >=36m and <60m, offshore = profile depth >=60m
The box limits extend from the lower to upper quartiles (25%, 75%), with a line at the median and a diamond symbol at
the mean. The whiskers extend from the box by 1.5x the inter-quartile range (IQR). Circles indicate outliers.
Notch indicates 95% CI around the median.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import copy
import matplotlib.pyplot as plt
import functions.common as cf
plt.rcParams.update({'font.size': 12})
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def build_summary(data, summary_dict, summary_dict_key):
    """
    :param data: data in the form of a numpy array
    :param summary_dict: dictionary to which summary data are appended
    :param summary_dict_key: key for dictionary
    :return:
    """
    if len(data) == 0:
        summary_dict[summary_dict_key] = dict(count=len(data),
                                              median=np.nan,
                                              mean=np.nan,
                                              lower_quartile=np.nan,
                                              upper_quartile=np.nan,
                                              lower_whisker=np.nan,
                                              upper_whisker=np.nan,
                                              min=np.nan,
                                              max=np.nan)
    else:
        lq = np.percentile(data, 25)
        uq = np.percentile(data, 75)
        iqr = uq - lq
        summary_dict[summary_dict_key] = dict(count=int(len(data)),
                                              median=np.round(np.nanmedian(data), 4),
                                              mean=np.round(np.nanmean(data), 4),
                                              lower_quartile=np.round(lq, 4),
                                              upper_quartile=np.round(uq, 4),
                                              lower_whisker=data[data >= lq - 1.5 * iqr].min(),
                                              upper_whisker=data[data <= uq + 1.5 * iqr].max(),
                                              min=np.round(np.nanmin(data), 4),
                                              max=np.round(np.nanmax(data), 4))


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


def main(fname, save_dir):
    ds = xr.open_dataset(fname)
    deploy = '-'.join(fname.split('/')[-1].split('-')[0:2])
    trsct = fname.split('/')[-1].split('-')[-1].split('.nc')[0]

    save_dir = os.path.join(save_dir, 'boxplots')
    os.makedirs(save_dir, exist_ok=True)

    # convert high oxygen values to nan
    oxyvars = [x for x in ds.data_vars if 'oxygen' in x]
    for ov in oxyvars:
        ds[ov][ds[ov] > 1000] = np.nan
        ds[ov][ds[ov] <= 0] = np.nan

    arr = np.array([], dtype='float32')
    divisions = dict(surface=dict(distance_bin=arr, values=arr, profile_depth=arr),
                     bottom=dict(distance_bin=arr, values=arr, profile_depth=arr))

    # initialize an empty data dictionary for each variable found in the dataset
    plt_vars = cf.plot_vars_glider()
    drop_vars = []
    for pv, info in plt_vars.items():
        try:
            variable_test = ds[pv]
            plt_vars[pv]['data'] = copy.deepcopy(divisions)
            plt_vars[pv]['summary'] = dict()
        except KeyError:
            drop_vars.append(pv)

    for dv in drop_vars:
        plt_vars.pop(dv)

    # initialize an empty array for the shelf location and MLD for each profile
    profile_dist_from_shore = np.array([])
    profile_mld_meters = np.array([])
    profile_max_depth = np.array([])
    profile_nomld_near = 0  # keep track of profiles that don't have MLD
    profile_nomld_mid = 0
    profile_nomld_off = 0

    # convert to dataframe and group by profile
    df = ds.to_dataframe()
    grouped = df.groupby('time')

    # for each profile, calculate 1m depth-binned averages, append the distance from shore and MLD variable
    for group in grouped:
        temp_df = cf.depth_bin(group[1])
        temp_df.dropna(subset=['depth'], inplace=True)
        profile_mld = np.nanmean(temp_df.mld)
        if np.isnan(profile_mld):  # if MLD isn't defined, throw out the profile
            if np.nanmax(temp_df.depth) < 36:
                profile_nomld_near += 1
            elif np.logical_and(np.nanmax(temp_df.depth) >= 36, np.nanmax(temp_df.depth) <= 60):
                profile_nomld_mid += 1
            elif np.nanmax(temp_df.depth) > 60:
                profile_nomld_off += 1
            continue

        profile_dist_from_shore = np.append(profile_dist_from_shore, np.round(np.nanmean(temp_df['dist_from_shore_km'])))
        profile_mld_meters = np.append(profile_mld_meters, profile_mld)

        # define the location on the shelf
        profile_maxz = np.nanmax(temp_df.depth)
        profile_max_depth = np.append(profile_max_depth, profile_maxz)

        # define the depth-bin and distance-bin the profile is in
        temp_df['depth_bin_1m'] = np.round(temp_df['depth'])
        temp_df['dist_from_shore_bin_1km'] = np.round(temp_df['dist_from_shore_km'])

        # separate surface and bottom waters
        surface_df = temp_df[temp_df.depth_bin_1m <= profile_mld]
        bottom_df = temp_df[temp_df.depth_bin_1m > profile_mld]

        for pv, info in plt_vars.items():
            dict_append_surface = info['data']['surface']
            dict_append_bottom = info['data']['bottom']
            dict_append_surface['distance_bin'] = np.append(dict_append_surface['distance_bin'],
                                                            surface_df['dist_from_shore_bin_1km'].values)
            dict_append_surface['values'] = np.append(dict_append_surface['values'], surface_df[pv].values)
            dict_append_surface['profile_depth'] = np.append(dict_append_surface['profile_depth'],
                                                             np.repeat(profile_maxz, len(surface_df[pv].values)))
            dict_append_bottom['distance_bin'] = np.append(dict_append_bottom['distance_bin'],
                                                           bottom_df['dist_from_shore_bin_1km'].values)
            dict_append_bottom['values'] = np.append(dict_append_bottom['values'], bottom_df[pv].values)
            dict_append_bottom['profile_depth'] = np.append(dict_append_bottom['profile_depth'],
                                                            np.repeat(profile_maxz, len(bottom_df[pv].values)))

    # boxplot of MLD
    # create empty dictionary for MLD
    mld_dict = dict(ttl='Mixed Layer Depth (m)',
                    data=dict(Nearshore=arr, Midshelf=arr, Offshore=arr))

    mld_df = pd.DataFrame(dict(dist_shore=profile_dist_from_shore,
                               mld=profile_mld_meters,
                               profile_depth=profile_max_depth))
    mld_df_binned = mld_df.groupby('dist_shore').mean()
    mld_dict['data']['Nearshore'] = np.array(mld_df_binned[mld_df_binned.profile_depth < 36]['mld'])
    mld_dict['data']['Midshelf'] = np.array(mld_df_binned[(mld_df_binned.profile_depth >= 36) & (mld_df_binned.profile_depth <= 60)]['mld'])
    mld_dict['data']['Offshore'] = np.array(mld_df_binned[mld_df_binned.profile_depth > 60]['mld'])

    bplot = []
    labels = []
    for k, v in mld_dict['data'].items():
        bplot.append(list(v))
        labels.append(f'{k}\nn={len(v)}')

    fig, ax = plt.subplots(figsize=(8, 9))

    # customize the boxplot elements
    medianprops = dict(color='black')
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black')
    boxprops = dict(facecolor='darkgray')

    box = ax.boxplot(bplot, patch_artist=True, labels=labels, showmeans=True, notch=True,
                     medianprops=medianprops, meanprops=meanpointprops, boxprops=boxprops, sym='.')
    ax.set_ylabel('Mixed Layer Depth (m)')

    ax.set_ylim([0, 22])
    ax.invert_yaxis()

    sfilename = f'{deploy}_boxplot_mld.png'
    sfile = os.path.join(save_dir, sfilename)
    plt.savefig(sfile, dpi=300)
    plt.close()

    box_colors = ['tab:blue', 'tab:orange', 'k']

    # initialize empty dictionary for summary to export
    summary_dict = dict()

    # iterate through each variable, bin data to 1-km distance, append summary data, and plot boxplots
    for pv, info in plt_vars.items():
        if 'oxygen' in pv:
            continue
        pv_dict = dict(surface=dict(Nearshore=arr, Midshelf=arr, Offshore=arr),
                       bottom=dict(Nearshore=arr, Midshelf=arr, Offshore=arr))
        for k, v in info['data'].items():
            pv_df = pd.DataFrame(dict(dist_shore=v['distance_bin'],
                                      data=v['values'],
                                      profile_depth=v['profile_depth']))
            pv_df_binned = pv_df.groupby('dist_shore').mean()
            pv_dict[k]['Nearshore'] = np.array(pv_df_binned[pv_df_binned.profile_depth < 36]['data'])
            pv_dict[k]['Midshelf'] = np.array(
                pv_df_binned[(pv_df_binned.profile_depth >= 36) & (pv_df_binned.profile_depth <= 60)]['data'])
            pv_dict[k]['Offshore'] = np.array(pv_df_binned[pv_df_binned.profile_depth > 60]['data'])

        nearsurf = pv_dict['surface']['Nearshore']
        midsurf = pv_dict['surface']['Midshelf']
        offsurf = pv_dict['surface']['Offshore']
        nearbot = pv_dict['bottom']['Nearshore']
        midbot = pv_dict['bottom']['Midshelf']
        offbot = pv_dict['bottom']['Offshore']

        # append summary data to dictionary
        summary_dict[pv] = dict()
        build_summary(nearsurf, summary_dict[pv], 'nearshore_surface')
        build_summary(midsurf, summary_dict[pv], 'midshelf_surface')
        build_summary(offsurf, summary_dict[pv], 'offshore_surface')
        build_summary(nearbot, summary_dict[pv], 'nearshore_bottom')
        build_summary(midbot, summary_dict[pv], 'midshelf_bottom')
        build_summary(offbot, summary_dict[pv], 'offshore_bottom')

        # make boxplot
        surface_data = [list(nearsurf),
                        list(midsurf),
                        list(offsurf)]
        bottom_data = [list(nearbot),
                       list(midbot),
                       list(offbot)]

        fig, ax = plt.subplots(figsize=(8, 9))

        # customize the boxplot elements
        meanpointprops = dict(marker='D')

        bp_surf = ax.boxplot(surface_data, positions=[1, 2, 3], widths=0.6, patch_artist=True, showmeans=True,
                             notch=True, meanprops=meanpointprops, sym='.')
        bp_bottom = ax.boxplot(bottom_data, positions=[5, 6, 7], widths=0.6, patch_artist=True, showmeans=True,
                               notch=True, meanprops=meanpointprops, sym='.')

        # set box colors
        set_box_colors(bp_surf, box_colors)
        set_box_colors(bp_bottom, box_colors)

        # draw temporary lines and use them to create a legend
        plt.plot([], c='tab:blue', label='Nearshore')
        plt.plot([], c='tab:orange', label='Midshelf')
        plt.plot([], c='k', label='Offshore')
        plt.legend(fontsize=12)

        # set axes labels
        ax.set_xticks([2, 6])
        ax.set_xticklabels(['Surface', 'Bottom'])
        ax.set_ylabel(info['ttl'])

        # draw a horizontal line between sections
        # ylims = ax.get_ylim()
        ylims = info['bplot_lims']
        ax.set_ylim(ylims)
        ax.vlines(4, ylims[0], ylims[1], colors='k')

        sfilename = f'{deploy}_boxplot_{pv}.png'
        sfile = os.path.join(save_dir, sfilename)
        plt.savefig(sfile, dpi=300)
        plt.close()

    # export summary as .csv
    df = pd.DataFrame()
    for k, v in summary_dict.items():
        dfk = pd.DataFrame(v)
        dfk.reset_index(inplace=True)
        dfk.index = list(np.repeat(k, len(dfk)))
        df = df.append(dfk)

    df = df.round(2)
    df = df.rename(columns={'index': 'statistic'})
    csv_filename = f'{deploy}_boxplot_summary.csv'
    csv_savefile = os.path.join(save_dir, 'summary_csv', csv_filename)
    df.to_csv(csv_savefile)


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/NOAA_OAP/gliderdata_dac/ru30-20190717T1812-delayed-transect1.nc'
    save_directory = '/Users/garzio/Documents/rucool/Saba/NOAA_OAP/wcr_analysis'
    main(ncfile, save_directory)
