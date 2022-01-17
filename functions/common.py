#! /usr/bin/env python3

import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import cmocean as cmo
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point


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


def extract_ecoa_data(lon_bounds, lat_bounds):
    """
    Extract ECOA data from CODAP-NA file
    """
    codap_file = '/Users/garzio/Documents/rucool/Saba/NOAA_SOE2021/data/CODAP_NA_v2021.nc'
    ds = xr.open_dataset(codap_file)
    codap_vars = dict(Cruise_ID=np.array([]),
                      Month_UTC=np.array([]),
                      Year_UTC=np.array([]),
                      Profile_number=np.array([]),
                      Latitude=np.array([]),
                      Longitude=np.array([]),
                      Depth=np.array([]),
                      Depth_bottom=np.array([]),
                      CTDTEMP_ITS90=np.array([]),
                      recommended_Salinity_PSS78=np.array([]),
                      pH_TS_insitu_calculated=np.array([]),
                      pH_TS_insitu_measured=np.array([]),
                      Aragonite=np.array([]),
                      TALK=np.array([]))

    # make sure the data are within the defined extent and find just the ECOA data
    for i, lon in enumerate(ds.Longitude.values):
        if Polygon(list(zip(lon_bounds, lat_bounds))).contains(Point(lon, ds.Latitude.values[i])):
            cid = ds.Cruise_ID.values[:, i]
            cid = [x.decode('UTF-8') for x in cid]
            cid = ''.join(cid).strip()
            if np.logical_or(cid == 'ECOA1', cid == 'ECOA2'):
                for key in codap_vars.keys():
                    if key == 'Cruise_ID':
                        codap_vars[key] = np.append(codap_vars[key], cid)
                    else:
                        codap_vars[key] = np.append(codap_vars[key], ds[key].values[i])

    df = pd.DataFrame(codap_vars)
    df['pH'] = df['pH_TS_insitu_measured']

    # use calculated pH if measured isn't available
    for idx, row in df.iterrows():
        if row['pH'] == -999:
            df.loc[row.name, 'pH'] = row.pH_TS_insitu_calculated

    return df


def glider_time_selection():
    glider_times = {'ru30-20210716T1804': {'first_transect': [dt.datetime(2021, 7, 16, 18, 0), dt.datetime(2021, 7, 24, 0, 0)]},
                    'sbu01-20210720T1628': {'first_transect': [dt.datetime(2021, 7, 20, 16, 0), dt.datetime(2021, 7, 29, 0, 0)],
                                            'last_transect': [dt.datetime(2021, 8, 8, 18, 0), dt.datetime(2021, 8, 18, 12, 0)]},
                    'ru30-20190717T1812': {'first_transect': [dt.datetime(2019, 7, 17, 16, 0), dt.datetime(2019, 7, 28, 0, 0)]}
                    }
    return glider_times


def glider_shore_locations():
    shore_locations = {'ru30-20210716T1804': {'first_transect': [39.48, -74.31]},  # Northern tip of Brigantine
                       'sbu01-20210720T1628': {'first_transect': [40.85, -72.5],
                                               'last_transect': [40.33, -73.98]},  # Long Branch, NJ (just south of Sandy Hook)
                       'ru30-20190717T1812': {'first_transect': [40.45, -74]}  # Sandy Hook
                       }
    return shore_locations


def profile_mld(df, mld_var='density', zvar='pressure', qi_threshold=0.5):
    """
    Written by Sam Coakley and Lori Garzio, Jan 2022
    Calculates the Mixed Layer Depth (MLD) for a single profile as the depth of max Brunt‐Vaisala frequency squared
    (N**2) from Carvalho et al 2016
    :param df: depth profile in the form of a pandas dataframe
    :param mld_var: the name of the variable for which MLD is calculated, default is 'density'
    :param zvar: the name of the depth variable in the dataframe, default is 'pressure'
    :param qi_threshold: quality index threshold for determining well-mixed water, default is 0.5
    :return: the depth of the mixed layer in the units of zvar
    """
    pN2 = np.sqrt(9.81 / np.nanmean(df[mld_var]) * np.diff(df[mld_var]) / np.diff(df[zvar])) ** 2
    if np.sum(~np.isnan(pN2)) == 0:
        mld = np.nan
    else:
        mld_idx = np.where(pN2 == np.nanmax(pN2))[0][0]
        mld = np.nanmean([df[zvar][mld_idx], df[zvar][mld_idx + 1]])

        if mld_idx == 0:
            # if the code finds the first data point as the MLD, return nan
            mld = np.nan
        elif mld < 2:
            # if MLD is <2, return nan
            mld = np.nan
        else:
            # find MLD  1.5
            mld15 = mld * 1.5
            mld15_idx = np.argmin(np.abs(df[zvar] - mld15))

            # Calculate Quality index (QI) from Lorbacher et al, 2006
            surface_mld_values = df[mld_var][0:mld_idx]  # values from the surface to MLD
            surface_mld15_values = df[mld_var][0:mld15_idx]  # values from the surface to MLD * 1.5

            qi = 1 - (np.std(surface_mld_values - np.nanmean(surface_mld_values)) /
                      np.std(surface_mld15_values - np.nanmean(surface_mld15_values)))

            if qi < qi_threshold:
                # if the Quality Index is < the threshold, this indicates well-mixed water so don't return MLD
                mld = np.nan

    return mld


def model_highlight_dates():
    md = {'ru30-20210716T1804': [[dt.datetime(2021, 7, 21, 12, 0), dt.datetime(2021, 7, 23, 12, 0)]],
          'sbu01-20211102T1528': [[dt.datetime(2021, 11, 5, 18, 0), dt.datetime(2021, 11, 6, 20, 0)],
                                  [dt.datetime(2021, 11, 6, 20, 0), dt.datetime(2021, 11, 8, 20, 0)]],
          'ru30-20210226T1647': [[dt.datetime(2021, 3, 6, 12, 0), dt.datetime(2021, 3, 8, 18, 0)]]}
    return md


def osm_mask_dates():
    md = {'ru30-20190717T1812': [[dt.datetime(2019, 7, 26, 6, 0), dt.datetime(2019, 7, 29, 18, 0)],
                                 [dt.datetime(2019, 8, 7, 12, 0), dt.datetime(2019, 8, 13, 0, 0)]],
          'ru30-20210716T1804': [[dt.datetime(2021, 7, 23, 12, 0), dt.datetime(2021, 7, 28, 6, 0)],
                                 [dt.datetime(2021, 8, 2, 0, 0), dt.datetime(2021, 8, 21, 0, 0)]],
          'sbu01-20210720T1628': [[dt.datetime(2021, 7, 28, 6, 0), dt.datetime(2021, 7, 30, 22, 0)],
                                  [dt.datetime(2021, 8, 1, 0, 0), dt.datetime(2021, 8, 3, 0, 0)],
                                  [dt.datetime(2021, 8, 5, 12, 0), dt.datetime(2021, 8, 10, 10, 0)],
                                  [dt.datetime(2021, 8, 18, 12, 0), dt.datetime(2021, 8, 21, 0, 0)]]}
    return md


def plot_vars():
    # plt_vars = {'conductivity': {'cmap': 'jet', 'ttl': 'Conductivity (S m-1)'},
    #             'temperature': {'cmap': cmo.cm.thermal, 'ttl': 'Temperature ({})'.format(r'$\rm ^oC$')},
    #             'salinity': {'cmap': cmo.cm.haline, 'ttl': 'Salinity'},
    #             'density': {'cmap': cmo.cm.dense, 'ttl': 'Density (kg m-3)'},
    #             'chlorophyll_a': {'cmap': cmo.cm.algae, 'ttl': 'Chlorophyll ({}g/L)'.format(chr(956))},
    #             'oxygen_concentration_mgL': {'cmap': cmo.cm.oxy, 'ttl': 'Oxygen (mg/L)'},
    #             'oxygen_concentration': {'cmap': cmo.cm.oxy, 'ttl': 'Oxygen (umol/L)'},
    #             'oxygen_concentration_shifted': {'cmap': cmo.cm.oxy, 'ttl': 'Oxygen (shifted) (umol/L)'},
    #             'sbe41n_ph_ref_voltage': {'cmap': cmo.cm.matter, 'ttl': 'pH Reference Voltage'},
    #             'sbe41n_ph_ref_voltage_shifted': {'cmap': cmo.cm.matter, 'ttl': 'pH Reference Voltage (shifted)'},
    #             'ph_total': {'cmap': cmo.cm.matter, 'ttl': 'pH'},
    #             'ph_total_shifted': {'cmap': cmo.cm.matter, 'ttl': 'pH (shifted)'}
    #             }
    plt_vars = {'temperature': {'cmap': cmo.cm.thermal, 'ttl': 'Temperature (\N{DEGREE SIGN}C)'},
                'chlorophyll_a': {'cmap': cmo.cm.algae, 'ttl': 'Chlorophyll ({}g/L)'.format(chr(956))},
                'sbe41n_ph_ref_voltage': {'cmap': cmo.cm.matter, 'ttl': 'pH Reference Voltage'}
                }
    return plt_vars


def plot_vars_glider():
    plt_vars = {'oxygen_concentration_mgL_shifted': {'cmap': cmo.cm.oxy, 'ttl': 'Oxygen (mg/L)',
                                                     'vmin': 5, 'vmax': 9, 'levels': np.arange(5, 9, 0.25),
                                                     'bplot_lims': [4, 11]},
                'temperature': {'cmap': cmo.cm.thermal, 'ttl': 'Temperature (\N{DEGREE SIGN}C)',
                                'vmin': 8, 'vmax': 26, 'levels': [8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
                                'bplot_lims': [6, 28]},
                'salinity': {'cmap': cmo.cm.haline, 'ttl': 'Salinity',
                             'vmin': 28, 'vmax': 36, 'levels': np.arange(29, 35, 0.5),
                             'bplot_lims': [28, 37]},
                'chlorophyll_a': {'cmap': cmo.cm.algae, 'ttl': 'Chlorophyll ({}g/L)'.format(chr(956)),
                                  'vmin': 0, 'vmax': 8, 'levels': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                  'bplot_lims': [0, 8]},
                'ph_total_shifted': {'cmap': cmo.cm.matter, 'ttl': 'pH',
                                     'vmin': 7.7, 'vmax': 8.3, 'levels': np.arange(7.7, 8.3, 0.05),
                                     'bplot_lims': [7.6, 8.3]},
                'total_alkalinity': {'cmap': cmo.cm.matter, 'ttl': 'Total Alkalinity',
                                     'vmin': 2000, 'vmax': 2400,
                                     'levels': [2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400],
                                     'bplot_lims': [1950, 2400]},
                'saturation_aragonite': {'cmap': cmo.cm.matter, 'ttl': 'Aragonite Saturation',
                                         'vmin': 1, 'vmax': 4.5, 'levels': np.arange(1, 4, 0.25),
                                         'bplot_lims': [.5, 4.5]}}
    return plt_vars


def plot_vars_ecoa():
    plt_vars = {'CTDTEMP_ITS90': {'cmap': cmo.cm.thermal, 'ttl': 'Temperature (\N{DEGREE SIGN}C)',
                                  'vmin': 8, 'vmax': 26},
                'recommended_Salinity_PSS78': {'cmap': cmo.cm.haline, 'ttl': 'Salinity',
                                               'vmin': 29, 'vmax': 35},
                'pH': {'cmap': cmo.cm.matter, 'ttl': 'pH',
                       'vmin': 7.7, 'vmax': 8.3},
                'TALK': {'cmap': cmo.cm.matter, 'ttl': 'Total Alkalinity',
                         'vmin': 2000, 'vmax': 2400},
                'Aragonite': {'cmap': cmo.cm.matter, 'ttl': 'Aragonite Saturation',
                              'vmin': 1, 'vmax': 4}}
    return plt_vars
