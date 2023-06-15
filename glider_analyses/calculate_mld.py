#!/usr/bin/env python

"""
Author: Lori Garzio on 1/11/2022
Last modified: 6/13/2023
Calculate Mixed Layer Depth for glider profiles and add the variable back into the .nc file
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions.common as cf
plt.rcParams.update({'font.size': 14})
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console


def main(fname, plots):
    ds = xr.open_dataset(fname)
    ds = ds.sortby(ds.time)

    df = ds.to_dataframe()
    grouped = df.groupby('time')
    mld = np.array([], dtype='float32')
    mldvar = 'density'
    zvar = 'pressure'

    if plots:
        plots = os.path.join(plots, 'mld_analysis')
        os.makedirs(plots, exist_ok=True)

    for group in grouped:
        if len(group[1].dropna(subset=[mldvar])) == 0:
            mldx = np.repeat(np.nan, len(group[1]))
        else:
            # Create temporary dataframe to interpolate to dz m depths
            kwargs = {'depth_var': zvar}
            temp_df = cf.depth_bin(group[1], **kwargs)
            temp_df.dropna(subset=[mldvar], inplace=True)
            if len(temp_df) == 0:
                mldx = np.repeat(np.nan, len(group[1]))
            else:
                # calculate profile's pressure range
                pressure_range = (np.nanmax(temp_df[zvar]) - np.nanmin(temp_df[zvar]))

                if pressure_range < 5:
                    # if the profile spans <5 dbar, don't calculate MLD
                    mldx = np.repeat(np.nan, len(group[1]))

                else:
                    mldx = cf.profile_mld(temp_df)
                    mldx = np.repeat(mldx, len(group[1]))

                    if plots:
                        try:
                            tstr = group[0].strftime("%Y-%m-%dT%H%M%SZ")
                        except AttributeError:
                            tstr = pd.to_datetime(np.nanmin(group[1].time)).strftime("%Y-%m-%dT%H%M%SZ")
                        # plot temperature
                        fig, ax = plt.subplots(figsize=(8, 10))
                        ax.scatter(temp_df['temperature'], temp_df[zvar])

                        ax.invert_yaxis()
                        ax.set_ylabel('Pressure (dbar)')
                        ax.set_xlabel('temperature')

                        ax.axhline(y=np.unique(mldx), ls='--', c='k')

                        sfile = os.path.join(plots, f'temperature_{tstr}.png')
                        plt.savefig(sfile, dpi=300)
                        plt.close()

                        # plot density
                        fig, ax = plt.subplots(figsize=(8, 10))
                        ax.scatter(temp_df['density'], temp_df[zvar])

                        ax.invert_yaxis()
                        ax.set_ylabel('Pressure (dbar)')
                        ax.set_xlabel('density')

                        ax.axhline(y=np.unique(mldx), ls='--', c='k')

                        sfile = os.path.join(plots, f'density{tstr}.png')
                        plt.savefig(sfile, dpi=300)
                        plt.close()

        mld = np.append(mld, mldx)

    # add mld to the dataset
    mld_min = np.nanmin(mld)
    mld_max = np.nanmax(mld)
    attrs = {
        'actual_range': np.array([mld_min, mld_max]),
        'ancillary_variables': [mldvar, zvar],
        'observation_type': 'calculated',
        'units': ds[zvar].units,
        'comment': 'Mixed Layer Depth calculated as the depth of max Brunt‐Vaisala frequency squared (N**2) from Carvalho et al 2016',
        'long_name': 'Mixed Layer Depth'
        }
    da = xr.DataArray(mld, coords=ds[mldvar].coords, dims=ds[mldvar].dims,
                      name='mld', attrs=attrs)
    ds['mld'] = da
    ds.to_netcdf(fname)


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/NOAA_OAP/gliderdata_dac/SBU01-20220805T1855-delayed.nc'
    generate_plots = False  # '/Users/garzio/Documents/rucool/Saba/NOAA_OAP/wcr_analysis'  # False
    main(ncfile, generate_plots)
