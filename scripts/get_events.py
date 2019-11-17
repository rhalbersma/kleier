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
from typing import Sequence, Tuple

import kleier.utils

def _event(eid: int, gid: int, table_header_rows: bs4.element.ResultSet) -> pd.DataFrame:
    name_place_date = (table_header_rows[0]
        .find('th')
        .text
        .split('\xa0\xa0\xa0\xa0\xa0\xa0')
    )
    name, place_date = ('', name_place_date[0]) if len(name_place_date) == 1 else name_place_date
    place_and_date = place_date.split()
    place = ' '.join(place_and_date[:-1])
    date = pd.to_datetime(place_and_date[-1])
    group, scoring = (table_header_rows[1]
        .find('th')
        .text
        .split('\xa0\xa0')
    )
    group = group.split(': ')[1][:-1]
    W, D, L = [
        int(points)
        for points in scoring.split(': ')[1].split()
    ]
    return pd.DataFrame(
        data   =[(eid,   gid,   name,   place,   date,   group,   W,   D,   L)],
        columns=['eid', 'gid', 'name', 'place', 'date', 'group', 'W', 'D', 'L']
    )

def _file(eid: int, results_from: str) -> Tuple[str]:
    return re.split(r'(^.*)\s?(\d{4}-\d{2}-\d{2})\s(.*$)', results_from)[1:-1]

def _tourn_table(eid: int, gid: int, table: bs4.element.Tag) -> pd.DataFrame:
    return (pd
        .read_html(str(table), header=[2, 3])[0]
        .assign(
            eid = eid,
            gid = gid
        )
        .pipe(lambda x: x.loc[:, x.columns.to_list()[-2:] + x.columns.to_list()[:-2]])
    )

def _standings(tourn_table: pd.DataFrame) -> pd.DataFrame:
    return tourn_table.drop(list(tourn_table.filter(regex='Results')), axis='columns')

def _results(tourn_table: pd.DataFrame) -> pd.DataFrame:
    return (pd.wide_to_long(tourn_table
                .filter(regex='eid|gid|#|Results')
                .pipe(lambda x: x.set_axis(x.columns.to_flat_index().map(''.join), axis='columns', inplace=False)),
            ['Results'], i='##', j='round'
        )
        .reset_index()
    )

def _event_group_table(eid: int, gid: int, table: bs4.element.Tag) -> Tuple[pd.DataFrame]:
    tourn_table = _tourn_table(eid, gid, table)
    last_row = tourn_table.tail(1)
    results_from, sep = str(last_row.iloc[0, 2]), 'Results from: '
    if results_from.startswith(sep):
        tourn_table.drop(last_row.index, inplace=True)
        file_from, file_date, file_name = _file(eid, results_from.split(sep)[1])
    else:
        file_from = file_date = file_name = ''
    event = (_event(eid, gid, table.find('thead').find_all('tr'))
        .assign(
            file_from = file_from,
            file_date = file_date,
            file_name = file_name
        )
    )
    standings = _standings(tourn_table)
    results = _results(tourn_table)
    return event, standings, results

def _download(eid: int) -> Tuple[pd.DataFrame]:
    assert 1 <= eid
    url = f'https://www.kleier.net/cgi/tourn_table.php?eid={eid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    tables = soup.find_all('table', {'summary': 'Stratego Tournament Cross-Table'})
    events, standings, results = tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _event_group_table(eid, gid, table)
            for gid, table in enumerate(tables)
        ])
    )
    remarks = pd.DataFrame(
        data=[
            (tables.index(remarks.find_previous('table')), remarks.text)
            for remarks in soup.find_all('pre')
        ],
        columns=['gid', 'remarks']
    )
    events = (pd
        .merge(events, remarks, how='outer')
        .fillna({'remarks': ''})
    )
    return events, standings, results

def _download_all(eids: Sequence[int]) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _download(eid)
            for eid in eids
        ])
    )

def format_standings(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .pipe(lambda x: x.set_axis(x.columns.to_flat_index().map('_'.join), axis='columns', inplace=False))
        .rename(columns=lambda x: x.strip('_'))
        .rename(columns=lambda x: re.sub(r'(.+)_\1', r'\1', x))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: x.replace('.', '_'))
        .rename(columns={
            '#': 'rank',
            'rating_value'    : 'Rnew',
            'rating_change'   : 'Rchg',
            'rating_eff_games': 'Reff_games'
        })
        .fillna({
            'surname': '',
            'prename': ''
        })
        .astype(dtype={column: int             for column in ['rank', 'standings_score']})
        .astype(dtype={column: float           for column in ['Rnew', 'Rchg']})
        .astype(dtype={column: pd.Int64Dtype() for column in ['Rnew', 'Rchg']})
        .astype(dtype={column: float           for column in ['Reff_games', 'standings_buchholz', 'standings_median']})
        .astype(dtype={'standings_compa': 'category'})
        .assign(Rold = lambda x: x.Rnew - x.Rchg)
        .loc[:, [
            'eid', 'gid',
            'rank', 'surname', 'prename', 'nationality',
            'Rold', 'Rchg', 'Rnew', 'Reff_games',
            'standings_score', 'standings_buchholz', 'standings_median', 'standings_compa'
        ]]
    )

def format_results(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .rename(columns={
            '##'     : 'rank1',
            'Results': 'result'
        })
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
        .loc[:, ['eid', 'gid', 'round', 'rank1', 'rank2', 'result']]
        .astype(dtype={column: int for column in ['rank1', 'rank2']})
        .astype(dtype={'result': 'category'})
    )

def main(max_eid: int) -> Tuple[pd.DataFrame]:
    events, standings, results = _download_all(range(1, 1 + max_eid))
    kleier.utils._save_dataset(events, 'events')
    kleier.utils._save_dataset(standings, 'standings')
    kleier.utils._save_dataset(results, 'results')
    return events, standings, results

if __name__ == '__main__':
    sys.exit(main(int(sys.argv[1])))
