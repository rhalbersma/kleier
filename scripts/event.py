#!/usr/bin/env python

#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re
import sys

import bs4
import pandas as pd
import requests

import kleier.utils

def _parse_group(eid, gid, table) -> tuple:
    event_table = (pd
        .read_html(str(table), header=3)[0]
        .assign(eid = eid, gid = gid)
        .rename(columns=lambda x: re.sub(r'(^\d+)', r'result\1', x))
    )
    last_row = event_table.tail(1)
    received = str(last_row.iloc[0, 0])
    results_from = 'Results from: '
    if received.startswith(results_from):
        event_table.drop(last_row.index, inplace=True)

    rounds = list(event_table.filter(regex='result\d+'))
    event_cross = (pd
        .wide_to_long(event_table.filter(['#', 'eid', 'gid'] + rounds), ['result'], i='#', j='round')
        .reset_index()
    )
    event_table.drop(columns=rounds, inplace=True)

    header_rows = table.find('thead').find_all('tr')
    name_place_date = header_rows[0].find('th').text
    group_scoring   = header_rows[1].find('th').text

    split = name_place_date.split('\xa0\xa0\xa0\xa0\xa0\xa0')
    name = '' if len(split) == 1 else split[0]
    place_date = split[0] if len(split) == 1 else split[1]
    split = place_date.split()
    place = ' '.join(split[:-1])
    date = pd.to_datetime(split[-1])

    group, scoring = group_scoring.split('\xa0\xa0')
    group = group.split(': ')[1][:-1]
    W, D, L = [
        int(points)
        for points in scoring.split(': ')[1].split()
    ]
    received = '' if not received.startswith(results_from) else received.split(results_from)[1]

    event_index = pd.DataFrame(
        data   =[(eid,   gid,   name,   place,   date,   group,   W,   D,   L,   received)],
        columns=['eid', 'gid', 'name', 'place', 'date', 'group', 'W', 'D', 'L', 'received']
    )

    return event_index, event_table, event_cross

def download(eid) -> tuple:
    url = f'https://www.kleier.net/cgi/tourn_table.php?eid={eid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    tables = soup.find_all('table', {'summary': 'Stratego Tournament Cross-Table'})
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _parse_group(eid, gid, table)
            for gid, table in enumerate(tables)
        ])
    )

def download_all(eids):
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            download(eid)
            for eid in eids
        ])
    )

def parse_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns.to_list()
    return (df
        .loc[:, columns[-2:] + columns[:-2]]
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: x.replace('.', '_'))
        .rename(columns={
            '#'          : 'rank',
            'surname'    : 'sur',
            'prename'    : 'pre',
            'nationality': 'nat',
            'value'      : 'rating'
        })
        .astype(dtype={column: int             for column in ['rank', 'score']})
        .astype(dtype={column: float           for column in ['rating', 'change']})
        .astype(dtype={column: pd.Int64Dtype() for column in ['rating', 'change']})
        .astype(dtype={column: float           for column in ['eff_games', 'buchholz', 'median']})
    )

def parse_cross(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .rename(columns={'#' : 'rank1'})
        .fillna({'result': '0-'})
        .replace(
            {'result': r'(\d+[+=-])[BW]'},
            {'result': r'\1'},
            regex=True
        )
        .assign(
            rank2  = lambda x: x.result.str.slice(0, -1),
            result = lambda x: x.result.str.slice(   -1)
        )
        .loc[:,['eid', 'gid', 'round', 'rank1', 'rank2', 'result']]
        .astype(dtype={column: int for column in ['rank1', 'rank2']})
    )

def main():
    event_index, event_table, event_cross = download_all(range(1, 1 + 662))
    kleier.utils._save_dataset(event_index, 'event_index')
    kleier.utils._save_dataset(event_table, 'event_table')
    kleier.utils._save_dataset(event_cross, 'event_cross')

if __name__ == '__main__':
    sys.exit(main())
