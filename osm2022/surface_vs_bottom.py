#!/usr/bin/env python

"""
Author: Lori Garzio on 1/19/2022
Last modified: 1/19/2022
Line plot of median surface vs bottom values as a function of distance from shore for glider and ECOA data.
1-m depth binned data for gliders.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import copy
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
import functions.common as cf
plt.rcParams.update({'font.size': 12})
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console


def main(file_list, save_dir, trsct, xlims):
    data = dict()
    arr = np.array([], dtype='float32')
    divisions = dict(distance_from_shore=arr,
                     surface_median=arr,
                     bottom_median=arr)
    for fname in file_list:
        ds = xr.open_dataset(fname)
        deploy = '-'.join(fname.split('/')[-1].split('-')[0:2])

        # select the transect
        time_range = cf.glider_time_selection()[deploy][trsct]
        ds = ds.sel(time=slice(time_range[0], time_range[1]))

        if deploy == 'ru30-20190717T1812':
            ds = ds.rename({'oxygen_mgl': 'oxygen_concentration_mgL_shifted', 'pH': 'ph_total_shifted',
                            'aragonite_saturation_state': 'saturation_aragonite'})

        for dov in ['oxygen_concentration', 'oxygen_concentration_shifted', 'oxygen_concentration_mgL',
                    'oxygen_concentration_mgL_shifted']:
            try:
                ds[dov].values[ds[dov].values <= 0] = np.nan
                ds[dov].values[ds[dov].values > 1000] = np.nan
            except KeyError:
                continue

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
                    plt_vars[pv]['data'] = copy.deepcopy(divisions)
            else:
                plt_vars[pv]['data'] = copy.deepcopy(divisions)
        for dv in drop_vars:
            new_dv = dv.split('_shifted')[0]
            plt_vars[new_dv] = plt_vars[dv]
            plt_vars[new_dv]['data'] = copy.deepcopy(divisions)
            plt_vars.pop(dv)

        # for each profile, append the distance from shore and variable for plotting
        for group in grouped:
            temp_df = cf.depth_bin(group[1])
            temp_df.dropna(subset=['depth'], inplace=True)
            profile_mld = np.nanmean(temp_df.mld_meters)
            if np.isnan(profile_mld):  # if MLD isn't defined, throw out the profile
                continue

            # separate surface and bottom waters
            surface_df = temp_df[temp_df.depth < profile_mld]
            bottom_df = temp_df[temp_df.depth > profile_mld]

            # append median surface/bottom data
            for pv, info in plt_vars.items():
                dict_append = info['data']
                dist_shore_append = np.nanmedian(temp_df['dist_from_shore_km'].values)
                surface_append = np.nanmedian(surface_df[pv].values)
                bottom_append = np.nanmedian(bottom_df[pv].values)
                dict_append['distance_from_shore'] = np.append(dict_append['distance_from_shore'], dist_shore_append)
                dict_append['surface_median'] = np.append(dict_append['surface_median'], surface_append)
                dict_append['bottom_median'] = np.append(dict_append['bottom_median'], bottom_append)

        # add the plt_vars dictionary to the main data dictionary for the deployment
        data[deploy] = plt_vars

    # generate plots
    plt_vars = cf.plot_vars_glider()
    for pv, info in plt_vars.items():
        for surfbot in ['surface_median', 'bottom_median']:
            fig, ax = plt.subplots(figsize=(12, 6))
            for deployment, values in data.items():
                if deployment == 'sbu01-20210720T1628':
                    color = 'tab:cyan'
                    label = 'sbu01-summer21'
                elif deployment == 'ru30-20190717T1812':
                    color = 'tab:orange'
                    label = 'ru30-summer19'
                elif deployment == 'ru30-20210716T1804':
                    color = 'k'
                    label = 'ru30-summer21'

                try:
                    pvdata = values[pv]['data']
                except KeyError:
                    pvdata = values[pv.split('_shifted')[0]]['data']

                ax.scatter(pvdata['distance_from_shore'], pvdata[surfbot], color=color, marker='.', s=6, label=label)

            ax.set_xlim(xlims)
            ax.set_ylim(info['lineplot_lims'])

            handles, labels = plt.gca().get_legend_handles_labels()  # only show one set of legend labels
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            # make legend points bigger
            for ha in ax.legend_.legendHandles:
                ha._sizes = [50]

            plt.setp(ax, ylabel=info['ttl'], xlabel='Distance from Shore (km)')

            sfile = os.path.join(save_dir, 'line_plots', f'{pv}_{surfbot}_vs_distfromshore.png')
            plt.savefig(sfile, dpi=200)
            plt.close()



if __name__ == '__main__':
    ncfiles = ['/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys_mld.nc',
               '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/sbu01-20210720T1628-profile-sci-delayed-qc_shifted_co2sys_mld.nc',
               '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20190717T1812-delayed_mld.nc']
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots'
    transect = 'first_transect'  # first_transect, last_transect (for sbu)
    x_limits = [0, 200]  # optional limits for x-axis  False
    main(ncfiles, save_directory, transect, x_limits)
