#!/usr/bin/env python

"""
Author: Lori Garzio on 1/12/2022
Last modified: 1/12/2022
Plot transects of ECOA data (distance from shore)
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from geographiclib.geodesic import Geodesic
import functions.common as cf
import functions.plotting as pf
plt.rcParams.update({'font.size': 12})
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console


def main(save_dir, transect):
    if transect == 'all':
        lon_bounds = [-77, -71, -71, -77]
        lat_bounds = [37.5, 37.5, 40.5, 40.5]
    elif transect == 'NJ':
        lon_bounds = [-74, -72, -72, -74]
        lat_bounds = [40, 38.8, 40.5, 40.5]
        shore_location = [40.45, -74]  # Sandy Hook
    elif transect == 'DE':
        lon_bounds = [-76, -73, -73, -75]
        lat_bounds = [39, 37.5, 37.9, 39]
        shore_location = [38.86, -75]  # mouth of DE Bay

    # get ECOA data from CODAP-NA
    df = cf.extract_ecoa_data(lon_bounds, lat_bounds)
    df = df[df.Depth <= 250]
    df = df.replace({-999: np.nan})
    years = np.unique(df.Year_UTC)

    # calculate distance from shore and add to dataframe
    df['dist_shore_km'] = ''
    geod = Geodesic.WGS84
    for i, row in df.iterrows():
        g = geod.Inverse(shore_location[0], shore_location[1], row.Latitude, row.Longitude)
        dist_km = g['s12'] * .001
        if row.Longitude < -75:  # point in the DE Bay
            dist_km = -dist_km
        df.loc[row.name, 'dist_shore_km'] = dist_km

    for yr in years:
        df_year = df[df.Year_UTC == yr]
        title = f'ECOA {transect} transect: {int(yr)}'

        # plot
        plt_vars = cf.plot_vars_ecoa()

        for pv, info in plt_vars.items():
            try:
                variable = df_year[pv]
            except KeyError:
                continue

            # plot xsection
            if len(variable) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                plt.subplots_adjust(left=0.08, right=0.92)
                kwargs = dict()
                kwargs['clabel'] = info['ttl']
                kwargs['title'] = title
                kwargs['xlabel'] = 'Distance from Shore (km)'
                kwargs['grid'] = True
                kwargs['cmap'] = info['cmap']
                kwargs['extend'] = 'neither'
                kwargs['markersize'] = 50
                kwargs['ylims'] = [0, 250]
                kwargs['vlims'] = [info['vmin'], info['vmax']]
                kwargs['edgecolor'] = 'k'
                pf.xsection(fig, ax, df_year.dist_shore_km.values, df_year.Depth.values, variable.values, **kwargs)

                sfile = os.path.join(save_dir, 'transects', 'ECOA', f'ECOA_{pv}_{transect}_transect_{int(yr)}.png')
                plt.savefig(sfile, dpi=200)
                plt.close()


if __name__ == '__main__':
    save_directory = '/Users/garzio/Documents/rucool/Saba/2021/NOAA_OAP/OSM2022/plots'
    transect = 'DE'  # 'NJ' 'DE'
    main(save_directory, transect)
