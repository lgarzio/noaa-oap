#!/usr/bin/env python

"""
Author: Lori Garzio on 1/19/2022
Last modified: 1/20/2022
Line plot of median surface vs bottom values as a function of distance from shore for glider and ECOA data.
1-m depth and 1-km distance binned data for gliders.
"""

import pickle
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


def main(file_list, save_dir, trsct, xlims, plt_ecoa_data, ac):
    data = dict()
    arr = np.array([], dtype='float32')
    divisions = dict(surface=dict(distance_bin=arr, values=arr),
                     bottom=dict(distance_bin=arr, values=arr))

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

        # convert to dataframe
        df = ds2.to_dataframe()

        # group each profile
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

        # initialize an empty array for distance from shore and MLD for each profile
        profile_dist_from_shore = np.array([])
        profile_mld_meters = np.array([])

        # for each profile, append the distance from shore and MLD variable
        for group in grouped:
            temp_df = cf.depth_bin(group[1])
            temp_df.dropna(subset=['depth'], inplace=True)
            profile_mld = np.nanmean(temp_df.mld_meters)
            if np.isnan(profile_mld):  # if MLD isn't defined, throw out the profile
                continue

            profile_dist_from_shore = np.append(profile_dist_from_shore, np.nanmean(temp_df['dist_from_shore_km']))
            profile_mld_meters = np.append(profile_mld_meters, profile_mld)

            # calculate 1-m depth and 1-km distance bins
            temp_df['depth_bin_1m'] = np.round(temp_df['depth'])
            temp_df['dist_from_shore_bin_1km'] = np.round(temp_df['dist_from_shore_km'])

            # separate surface and bottom waters
            surface_df = temp_df[temp_df.depth < profile_mld]
            bottom_df = temp_df[temp_df.depth > profile_mld]

            for pv, info in plt_vars.items():
                dict_append_surface = info['data']['surface']
                dict_append_bottom = info['data']['bottom']
                dict_append_surface['distance_bin'] = np.append(dict_append_surface['distance_bin'],
                                                                surface_df['dist_from_shore_bin_1km'].values)
                dict_append_surface['values'] = np.append(dict_append_surface['values'], surface_df[pv].values)
                dict_append_bottom['distance_bin'] = np.append(dict_append_bottom['distance_bin'],
                                                               bottom_df['dist_from_shore_bin_1km'].values)
                dict_append_bottom['values'] = np.append(dict_append_bottom['values'], bottom_df[pv].values)

        # add the plt_vars dictionary to the main data dictionary for the deployment
        data[deploy] = plt_vars

        # # write pickle
        # with open('glider_data.pickle', 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # ECOA data
    ecoa_data = dict(NJ=dict(yr2015=dict(), yr2018=dict()),
                     DE=dict(yr2015=dict(), yr2018=dict()))
    for ecoa_trsct in ['NJ', 'DE']:
        lon_bounds, lat_bounds = cf.ecoa_transect_bounds(ecoa_trsct)
        shore_location = cf.ecoa_transect_shore_point(ecoa_trsct)

        # get ECOA data from CODAP-NA
        df = cf.extract_ecoa_data(lon_bounds, lat_bounds)
        df = df[df.Depth <= 250]
        df = df.replace({-999: np.nan})

        # calculate density - have to calculate Absolute Salinity (SA) and Conservative Temperature (CT) first
        sa = gsw.SA_from_SP(df.recommended_Salinity_PSS78, df.CTDPRES, df.Longitude, df.Latitude)
        ct = gsw.CT_from_t(sa, df.CTDTEMP_ITS90, df.CTDPRES)
        df['density'] = gsw.density.rho(sa, ct, df.CTDPRES)

        for yr in [2015, 2018]:
            # initialize empty data dictionary for each transect/year
            ecoa_vars = cf.plot_vars_ecoa()
            for ev, values in ecoa_vars.items():
                values['data'] = dict(dist_shore=np.array([]),
                                      surface=np.array([]),
                                      bottom=np.array([]))

            dfyr = df[df.Year_UTC == yr]

            # calculate MLD and distance from shore for each profile
            kwargs = dict()
            kwargs['zvar'] = 'CTDPRES'
            kwargs['qi_threshold'] = None
            binargs = dict()
            binargs['depth_var'] = 'CTDPRES'
            for pn in np.unique(dfyr.Profile_number):
                dfpn = dfyr[dfyr.Profile_number == pn]
                temp_df = cf.depth_bin(dfpn, **binargs)
                temp_df.dropna(subset=['CTDPRES'], inplace=True)
                mld = cf.profile_mld(temp_df, **kwargs)
                if np.isnan(mld):
                    continue

                # calculate distance from shore
                temp_df['dist_shore_km'] = ''
                geod = Geodesic.WGS84
                for i, row in temp_df.iterrows():
                    g = geod.Inverse(shore_location[0], shore_location[1], row.Latitude, row.Longitude)
                    dist_km = g['s12'] * .001
                    if row.Longitude < -75:  # point in the DE Bay
                        dist_km = -dist_km
                    temp_df.loc[row.name, 'dist_shore_km'] = dist_km
                pn_dist_shore_km = np.round(np.nanmean(list((temp_df['dist_shore_km']))))

                temp_df_surface = temp_df[temp_df.CTDPRES < mld]
                temp_df_bottom = temp_df[temp_df.CTDPRES > mld]

                # append data for each variable to dictionary
                for ev, values in ecoa_vars.items():
                    values['data']['dist_shore'] = np.append(values['data']['dist_shore'], pn_dist_shore_km)
                    values['data']['surface'] = np.append(values['data']['surface'], np.nanmedian(temp_df_surface[ev].values))
                    values['data']['bottom'] = np.append(values['data']['bottom'], np.nanmedian(temp_df_bottom[ev].values))

            ecoa_data[ecoa_trsct][f'yr{yr}'] = ecoa_vars

    # with open('glider_data.pickle', 'rb') as handle:
    #     data = pickle.load(handle)

    # generate plots
    plt_vars = cf.plot_vars_glider()
    for pv, info in plt_vars.items():
        if pv == 'temperature':
            ecoav = 'CTDTEMP_ITS90'
        elif pv == 'salinity':
            ecoav = 'recommended_Salinity_PSS78'
        elif pv in ['ph_total_shifted', 'ph_total']:
            ecoav = 'pH'
        elif pv == 'total_alkalinity':
            ecoav = 'TALK'
        elif pv == 'saturation_aragonite':
            ecoav = 'Aragonite'
        else:
            ecoav = None

        for surfbot in ['surface', 'bottom']:
            fig, ax = plt.subplots(figsize=(12, 6))

            # plot glider data
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
                    pvdata = values[pv]['data'][surfbot]
                except KeyError:
                    pvdata = values[pv.split('_shifted')[0]]['data'][surfbot]

                # for each 1km distance bin, calculate the median value
                dist_shore = np.array([])
                median_values = np.array([])
                for db in np.unique(pvdata['distance_bin']):
                    dist_shore = np.append(dist_shore, db)
                    db_idx = np.where(pvdata['distance_bin'] == db)[0]
                    median_values = np.append(median_values, np.nanmedian(pvdata['values'][db_idx]))

                ax.scatter(dist_shore, median_values, color=color, marker='.', s=25, label=label)

                # add a vertical line where the 10C isobath intersects with the bottom to indicate the extent of the coldpool
                ax.vlines(cp[deployment][trsct], info['lineplot_lims'][0], info['lineplot_lims'][1], colors=color,
                          linestyles='--')

            # plot ECOA data
            if plt_ecoa_data:
                if ecoav:
                    ecolors = ['purple', 'blue']
                    emarkers = ['D', 'x']
                    for i, ecoa_trsct in enumerate(['NJ', 'DE']):
                        for j, yr in enumerate([2015, 2018]):
                            lab = f'ECOA-{ecoa_trsct}{yr}'
                            ecoa_plot_data = ecoa_data[ecoa_trsct][f'yr{yr}'][ecoav]['data']
                            ax.scatter(ecoa_plot_data['dist_shore'], ecoa_plot_data[surfbot], color=ecolors[i],
                                       marker=emarkers[j], s=15, label=lab)

            ax.set_xlim(xlims)
            ax.set_ylim(info['lineplot_lims'])

            handles, labels = plt.gca().get_legend_handles_labels()  # only show one set of legend labels
            by_label = dict(zip(labels, handles))
            if plt_ecoa_data:
                plt.legend(by_label.values(), by_label.keys(), fontsize=10, ncol=3)
            else:
                plt.legend(by_label.values(), by_label.keys(), fontsize=10)

            # make legend points bigger
            # for ha in ax.legend_.legendHandles:
            #     ha._sizes = [70]

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
    plot_ecoa = True
    add_coldpool = True
    main(ncfiles, save_directory, transect, x_limits, plot_ecoa, add_coldpool)
