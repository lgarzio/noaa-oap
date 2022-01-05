#! /usr/bin/env python3

import datetime as dt


def mask_dates():
    md = {'ru30-summer2019': [[dt.datetime(2019, 7, 26, 6, 0), dt.datetime(2019, 7, 29, 18, 0)],
                              [dt.datetime(2019, 8, 7, 12, 0), dt.datetime(2019, 8, 13, 0, 0)]],
          'ru30-summer2021': [[dt.datetime(2021, 7, 23, 12, 0), dt.datetime(2021, 7, 28, 6, 0)],
                              [dt.datetime(2021, 8, 2, 0, 0), dt.datetime(2021, 8, 21, 0, 0)]],
          'sbu01-summer2021': [[dt.datetime(2021, 7, 28, 6, 0), dt.datetime(2021, 7, 30, 22, 0)],
                               [dt.datetime(2021, 8, 1, 0, 0), dt.datetime(2021, 8, 3, 0, 0)],
                               [dt.datetime(2021, 8, 5, 12, 0), dt.datetime(2021, 8, 10, 10, 0)],
                               [dt.datetime(2021, 8, 18, 12, 0), dt.datetime(2021, 8, 21, 0, 0)]]}
    return md
