#!/usr/bin/env python

"""
Author: Lori Garzio on 1/21/2022
Last modified: 1/24/2022
Calculate chl-a maximum and 35m integrated chlorophyll for glider profiles. If profile range is <10m, returns nan.
Calculates integrated chlorophyll if profile range is >10m even if max profile depth is <35m. Adds the variables back
to the original .nc file.
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions.common as cf
plt.rcParams.update({'font.size': 14})
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console


def main(fname, plots, integratez):
    ds = xr.open_dataset(fname)
    ds = ds.sortby(ds.time)
    ds2 = ds.swap_dims({'time': 'profile_time'})

    df = ds2.to_dataframe()
    grouped = df.groupby('profile_time')
    integrated_chl = np.array([], dtype='float32')
    chlorophyll_max = np.array([], dtype='float32')
    chlvar = 'chlorophyll_a'
    zvar = 'depth'

    for group in grouped:
        # Create temporary dataframe to interpolate to dz m depths
        kwargs = {'depth_var': zvar}
        temp_df = cf.depth_bin(group[1], **kwargs)
        temp_df.dropna(subset=[chlvar], inplace=True)
        if len(temp_df) == 0:
            chlmax = np.repeat(np.nan, len(group[1]))
            ichl = np.repeat(np.nan, len(group[1]))
        else:
            # calculate profile's pressure range
            pressure_range = (np.nanmax(temp_df[zvar]) - np.nanmin(temp_df[zvar]))

            if pressure_range < 10:
                # if the profile spans <10 dbar, don't calculate chlmax
                chlmax = np.repeat(np.nan, len(group[1]))
                ichl = np.repeat(np.nan, len(group[1]))

            else:
                # calculate chlorophyll max
                chlm = cf.profile_chl_max(temp_df)
                chlmax = np.repeat(chlm, len(group[1]))

                # calculate depth-integrated chlorophyll
                chl_int = cf.profile_integrated(temp_df, integratez)
                ichl = np.repeat(chl_int, len(group[1]))

                if plots:
                    try:
                        tstr = group[0].strftime("%Y-%m-%dT%H%M%SZ")
                    except AttributeError:
                        tstr = pd.to_datetime(np.nanmin(group[1].time)).strftime("%Y-%m-%dT%H%M%SZ")
                    # plot chl
                    fig, ax = plt.subplots(figsize=(8, 10))
                    ax.scatter(temp_df[chlvar], temp_df[zvar])

                    ax.invert_yaxis()
                    ax.set_ylabel('Depth (m)')
                    ax.set_xlabel('chl-a')

                    ax.axhline(y=np.unique(chlm), ls='--', c='k')

                    sfile = os.path.join(plots, f'chla_{tstr}.png')
                    plt.savefig(sfile, dpi=300)
                    plt.close()

        chlorophyll_max = np.append(chlorophyll_max, chlmax)
        integrated_chl = np.append(integrated_chl, ichl)

    # add chl max to the dataset
    chlorophyll_max_min = np.nanmin(chlorophyll_max)
    chlorophyll_max_max = np.nanmax(chlorophyll_max)
    attrs = {
        'actual_range': np.array([chlorophyll_max_min, chlorophyll_max_max]),
        'ancillary_variables': f'{chlvar} {zvar}',
        'observation_type': 'calculated',
        'units': ds[zvar].units,
        'comment': 'Chlorophyll-a Maximum Depth for each profile calculated as the depth of the maximum 1-m depth-binned chlorophyll-a value',
        'long_name': 'Chlorophyll-a Maximum Depth'
        }
    da = xr.DataArray(chlorophyll_max, coords=ds[chlvar].coords, dims=ds[chlvar].dims,
                      name='chlorophyll_max', attrs=attrs)
    ds['chlorophyll_max'] = da

    # add integrated chl to the dataset
    int_chl_min = np.nanmin(integrated_chl)
    int_chl_max = np.nanmax(integrated_chl)
    attrs = {
        'actual_range': np.array([int_chl_min, int_chl_max]),
        'ancillary_variables': f'{chlvar} {zvar}',
        'observation_type': 'calculated',
        'units': 'mg m-2',
        'comment': f'{integratez}m depth-integrated chlorophyll-a for each profile',
        'long_name': 'Integrated Chlorophyll-a'
    }
    da = xr.DataArray(integrated_chl, coords=ds[chlvar].coords, dims=ds[chlvar].dims,
                      name='integrated_chl', attrs=attrs)
    ds['integrated_chl'] = da
    ds.to_netcdf(f'{fname.split(".nc")[0]}_chl_analysis.nc')


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys_mld.nc'
    # ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/sbu01-20210720T1628-profile-sci-delayed-qc_shifted_co2sys_mld.nc'
    # ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20190717T1812-delayed_mld.nc'
    generate_plots = False  # '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/chlmax/ru30_2021'  # False
    integrate_depth = 35
    main(ncfile, generate_plots, integrate_depth)
