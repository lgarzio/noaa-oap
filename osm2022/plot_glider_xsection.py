#!/usr/bin/env python

"""
Author: Lori Garzio on 1/14/2022
Last modified: 1/142022
Plot first cross-shelf transect for each glider deployment (scatter plots)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console


def main(fname, save_dir, plt_mld, trsct):
    ds = xr.open_dataset(fname)
    deploy = '-'.join(fname.split('/')[-1].split('-')[0:2])

    # select the first cross-shelf transect
    time_range = cf.glider_time_selection()[deploy][trsct]
    ds = ds.sel(time=slice(time_range[0], time_range[1]))

    if deploy == 'ru30-20190717T1812':
        ds = ds.rename({'oxygen_mgl': 'oxygen_concentration_mgL_shifted', 'pH': 'ph_total_shifted',
                        'aragonite_saturation_state': 'saturation_aragonite'})

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

    xmin = np.floor(np.nanmin(ds.dist_from_shore_km))
    xmax = np.ceil(np.nanmax(ds.dist_from_shore_km))

    # for each variable figure out if any of the variables were time-shifted
    plt_vars = cf.plot_vars_glider()
    drop_vars = []
    for pv, info in plt_vars.items():
        if '_shifted' in pv:  # if the data weren't time shifted, take the non-shifted variable
            if np.sum(~np.isnan(ds[pv].values)) == 0:
                drop_vars.append(pv)

    for dv in drop_vars:
        new_dv = dv.split('_shifted')[0]
        plt_vars[new_dv] = plt_vars[dv]
        plt_vars.pop(dv)

    # iterate through each variable, turn the data into a binned dataframe and generate a contourf plot
    for pv, info in plt_vars.items():
        variable = ds[pv]
        extend = 'both'
        if 'chlorophyll' in pv:
            extend = 'max'

        # remove bad DO values
        if 'oxygen' in pv:
            variable[variable > 1000] = np.nan

        # plot xsection
        if len(variable) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.subplots_adjust(left=0.08, right=0.92)
            figttl_xsection = f'{deploy} {info["ttl"].split(" (")[0]}'
            kwargs = dict()
            kwargs['clabel'] = info['ttl']
            kwargs['title'] = figttl_xsection
            kwargs['grid'] = True
            kwargs['cmap'] = info['cmap']
            kwargs['extend'] = extend
            kwargs['xlabel'] = 'Distance from Shore (km)'
            kwargs['xlims'] = [xmin, xmax]
            kwargs['vlims'] = [info['vmin'], info['vmax']]
            pf.xsection(fig, ax, ds.dist_from_shore_km.values, ds.depth.values, variable.values, **kwargs)

            if plt_mld:
                ax.plot(ds.dist_from_shore_km.values, ds.mld_meters.values, ls='-', lw=1.5, color='k')

            sfilename = f'{deploy}_xsection_{trsct}_{pv}.png'
            sfile = os.path.join(save_dir, 'xsection', sfilename)
            plt.savefig(sfile, dpi=300)
            plt.close()


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys_mld.nc'
    # ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/sbu01-20210720T1628-profile-sci-delayed-qc_shifted_co2sys_mld.nc'
    # ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20190717T1812-delayed_mld.nc'
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots'
    plot_mld = 'True'  # True False
    transect = 'last_transect'  # first_transect, last_transect (for sbu)
    main(ncfile, save_directory, plot_mld, transect)
