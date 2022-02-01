#!/usr/bin/env python

"""
Author: Lori Garzio on 1/13/2022
Last modified: 1/19/2022
Plot first cross-shelf transect for each glider deployment (1-m depth binned 1-km distance binned contour plot)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
import functions.common as cf
plt.rcParams.update({'font.size': 12})
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console


def main(fname, save_dir, plt_mld, trsct, ymax, xlims, ac):
    ds = xr.open_dataset(fname)
    deploy = '-'.join(fname.split('/')[-1].split('-')[0:2])

    # select the transect
    time_range = cf.glider_time_selection()[deploy][trsct]
    ds = ds.sel(time=slice(time_range[0], time_range[1]))

    if deploy == 'ru30-20190717T1812':
        ds = ds.rename({'oxygen_mgl': 'oxygen_concentration_mgL_shifted', 'pH': 'ph_total_shifted',
                        'aragonite_saturation_state': 'saturation_aragonite'})

    depth_max = np.ceil(np.nanmax(ds.depth.values))

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

    # get the shore location for the first cross-shelf transect
    shore_location = cf.glider_shore_locations()[deploy][trsct]

    # calculate distance from shore
    geod = Geodesic.WGS84
    dist_from_shore_km = np.array(np.empty(len(ds.profile_lat)))
    dist_from_shore_km[:] = np.nan
    profile_lats = np.unique(ds.profile_lat)
    for pl in profile_lats:
        pl_idx = np.where(ds.profile_lat.values == pl)[0]
        plon = ds.profile_lon.values[pl_idx]
        g = geod.Inverse(shore_location[0], shore_location[1], pl, np.unique(plon)[0])
        dist_km = g['s12'] * .001
        dist_from_shore_km[pl_idx] = dist_km

    # add distance from shore to dataset
    attrs = {
        'actual_range': np.array([np.nanmin(dist_from_shore_km), np.nanmax(dist_from_shore_km)]),
        'observation_type': 'calculated',
        'units': 'km',
        'comment': f'distance from shore, calculated from shore location {shore_location}',
        'long_name': 'Distance from Shore'
    }
    da = xr.DataArray(dist_from_shore_km, coords=ds.profile_lat.coords, dims=ds.profile_lat.dims,
                      name='dist_from_shore_km', attrs=attrs)
    ds['dist_from_shore_km'] = da

    ds2 = ds.swap_dims({'time': 'profile_time'})

    # convert to dataframe and group by profile
    df = ds2.to_dataframe()
    grouped = df.groupby('profile_time')

    # for each variable figure out if any of the variables were time-shifted and initialize an empty list to append data
    plt_vars = cf.plot_vars_glider()
    drop_vars = []
    for pv, info in plt_vars.items():
        if '_shifted' in pv:  # if the data weren't time shifted, take the non-shifted variable
            if np.sum(~np.isnan(ds[pv].values)) == 0:
                drop_vars.append(pv)
            else:
                plt_vars[pv]['data'] = []
        else:
            plt_vars[pv]['data'] = []
    for dv in drop_vars:
        new_dv = dv.split('_shifted')[0]
        plt_vars[new_dv] = plt_vars[dv]
        plt_vars[new_dv]['data'] = []
        plt_vars.pop(dv)

    # initialize an empty array for distance from shore and MLD for each profile for plotting
    profile_dist_from_shore = np.array([])
    profile_mld_meters = np.array([])

    # for each profile, append the distance from shore and the variable data for each variable for plotting
    for group in grouped:
        kwargs = {'depth_max': depth_max}
        temp_df = cf.depth_bin(group[1], **kwargs)
        profile_dist_from_shore = np.append(profile_dist_from_shore, np.nanmean(temp_df['dist_from_shore_km']))
        profile_mld_meters = np.append(profile_mld_meters, np.nanmean(temp_df['mld_meters']))

        # add data to each variable for each profile
        for pv, info in plt_vars.items():
            info['data'].append(temp_df[pv].values.tolist())

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
        cp = cf.glider_coldpool_extent()[deploy][trsct]
        cpdf = temperature_df[cp].dropna()
        cp_depth = np.nanmax(cpdf.index)

    # iterate through each variable, turn the data into a binned dataframe and generate a contourf plot
    for pv, info in plt_vars.items():
        extend = 'both'
        if 'chlorophyll' in pv:
            extend = 'max'

        pv_df = pd.DataFrame(info['data'])

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

        sfile = os.path.join(save_dir, 'transects', 'glider', f'{deploy}_{trsct}_{pv}.png')
        plt.savefig(sfile, dpi=200)
        plt.close()


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys_mld.nc'
    # ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/sbu01-20210720T1628-profile-sci-delayed-qc_shifted_co2sys_mld.nc'
    # ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20190717T1812-delayed_mld.nc'
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots'
    plot_mld = 'True'  # True False
    transect = 'first_transect'  # first_transect, last_transect (for sbu)
    ymaximum = 180  # max depth for y-axis  False
    x_limits = [10, 200]  # optional limits for x-axis  False
    add_coldpool = True
    main(ncfile, save_directory, plot_mld, transect, ymaximum, x_limits, add_coldpool)
