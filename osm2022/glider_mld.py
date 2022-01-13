#!/usr/bin/env python

"""
Author: Lori Garzio on 1/11/2022
Last modified: 1/12/2022
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


def depth_bin(dataframe, depth_var='depth', depth_min=0, depth_max=None, stride=1):
    """
    Written by Mike Smith
    :param dataframe: depth profile in the form of a pandas dataframe
    :param depth_var: the name of the depth variable in the dataframe
    :param depth_min: the shallowest bin depth
    :param depth_max: the deepest bin depth
    :param stride: the amount of space between each bin
    :return: pandas dataframe where data has been averaged into specified depth bins
    """
    depth_max = depth_max or dataframe[depth_var].max()

    bins = np.arange(depth_min, depth_max+stride, stride)  # Generate array of depths you want to bin at
    cut = pd.cut(dataframe[depth_var], bins)  # Cut/Bin the dataframe based on the bins variable we just generated
    binned_df = dataframe.groupby(cut).mean()  # Groupby the cut and do the mean
    return binned_df


def main(fname, plots):
    ds = xr.open_dataset(fname)
    ds = ds.sortby(ds.time)
    try:
        ds2 = ds.swap_dims({'time': 'profile_time'})
    except ValueError:  # this is for the dataset downloaded from the DAC
        ds = ds.drop_dims({'trajectory'})
        profile_time = np.array([], dtype='datetime64[ns]')
        profile_lat = np.array([])
        profile_lon = np.array([])
        for rs in ds.rowSize:
            new_time = np.repeat(rs.time.values, rs.values)
            new_lat = np.repeat(rs.latitude.values, rs.values)
            new_lon = np.repeat(rs.latitude.values, rs.values)
            profile_time = np.append(profile_time, new_time)
            profile_lat = np.append(profile_lat, new_lat)
            profile_lon = np.append(profile_lon, new_lon)

        # add variables to dataset
        da = xr.DataArray(profile_time, coords=ds.conductivity.coords, dims=ds.conductivity.dims,
                          name='profile_time')
        ds['profile_time'] = da
        da = xr.DataArray(profile_time, coords=ds.conductivity.coords, dims=ds.conductivity.dims,
                          name='time')
        attrs = {
            'comment': 'same as profile_time',
        }
        ds['time'] = da
        da = xr.DataArray(profile_lat, coords=ds.conductivity.coords, dims=ds.conductivity.dims,
                          name='profile_lat', attrs=attrs)
        ds['profile_lat'] = da
        da = xr.DataArray(profile_lon, coords=ds.conductivity.coords, dims=ds.conductivity.dims,
                          name='profile_lon')
        ds['profile_lon'] = da
        ds = ds.drop_dims({'profile'})
        ds = ds.swap_dims({'obs': 'time'})
        ds2 = ds.swap_dims({'time': 'profile_time'})

    df = ds2.to_dataframe()
    grouped = df.groupby('profile_time')
    mld = np.array([], dtype='float32')
    mldvar = 'density'
    zvar = 'pressure'

    for group in grouped:
        # Create temporary dataframe to interpolate to dz m depths
        kwargs = {'depth_var': zvar}
        temp_df = depth_bin(group[1], **kwargs)
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
    ds.to_netcdf(f'{fname.split(".nc")[0]}_mld.nc')


if __name__ == '__main__':
    ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20210716T1804-profile-sci-delayed-qc_shifted_co2sys.nc'
    # ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/sbu01-20210720T1628-profile-sci-delayed-qc_shifted_co2sys.nc'
    # ncfile = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/data/ru30-20190717T1812-delayed.nc'
    generate_plots = False  # '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/mld_ru30_2019'  # False
    main(ncfile, generate_plots)
