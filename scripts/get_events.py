#!/usr/bin/env python

#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re
import sys

import bs4
import numpy as np
import pandas as pd
import requests
from typing import Sequence, Tuple

import kleier.utils

def _event_group(eid: int, gid: int, table_header_rows: bs4.element.ResultSet) -> pd.DataFrame:
    name_place_date = (table_header_rows[0]
        .find('th')
        .text
        .split('\xa0\xa0\xa0\xa0\xa0\xa0')
    )
    name, place_date = (np.nan, name_place_date[0]) if len(name_place_date) == 1 else name_place_date
    place_and_date = place_date.split()
    place = ' '.join(place_and_date[:-1])
    date = pd.to_datetime(place_and_date[-1])
    group, scoring = (table_header_rows[1]
        .find('th')
        .text
        .split('\xa0\xa0')
    )
    group = group.split(': ')[1][:-1]
    pW, pD, pL = [
        int(points)
        for points in scoring.split(': ')[1].split()
    ]
    return pd.DataFrame(
        data   =[(eid,   gid,   name,   place,   date,   group,   pW,   pD,   pL)],
        columns=['eid', 'gid', 'name', 'place', 'date', 'group', 'pW', 'pD', 'pL']
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

def _unplayed(M: int, N: int, table: bs4.element.Tag) -> pd.DataFrame:
    return pd.DataFrame(
        data = [[
                False if not td.has_attr('class') else td['class'][0] == 'unplayed'
                for td in tr.find_all('td')[-N:]
            ]
            for tr in table.find_all('tr')[4:4+M]
        ],
        columns = pd.MultiIndex.from_tuples([
            ('Unplayed', str(n + 1))
            for n in range(N)
        ])
    )

def _standings(tourn_table: pd.DataFrame) -> pd.DataFrame:
    return tourn_table.drop(list(tourn_table.filter(regex='Results|Unplayed')), axis='columns')

def _results(tourn_table: pd.DataFrame) -> pd.DataFrame:
    return (pd.wide_to_long(tourn_table
                .filter(regex='eid|gid|#|Results|Unplayed')
                .pipe(lambda x: x
                    .set_axis(x
                        .columns
                        .to_flat_index()
                        .map(''.join)
                        , axis='columns', inplace=False
                    )
                ),
            ['Results', 'Unplayed'], i='##', j='round'
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
        file_from = file_date = file_name = np.nan
    M, N = tourn_table.filter(regex='Results').shape
    tourn_table = tourn_table.join(_unplayed(M, N, table))
    event_group = (_event_group(eid, gid, table.find('thead').find_all('tr'))
        .assign(
            M = M,
            N = N,
            file_from = file_from,
            file_date = file_date,
            file_name = file_name
        )
    )
    standings = _standings(tourn_table)
    results = _results(tourn_table)
    return event_group, standings, results

def _download(eid: int) -> Tuple[pd.DataFrame]:
    assert 1 <= eid
    url = f'https://www.kleier.net/cgi/tourn_table.php?eid={eid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    tables = soup.find_all('table', {'summary': 'Stratego Tournament Cross-Table'})
    remarks = pd.DataFrame(
        data=[
            (tables.index(remarks.find_previous('table')), remarks.text)
            for remarks in soup.find_all('pre')
        ],
        columns=['gid', 'remarks']
    )
    event_groups, standings, results = tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _event_group_table(eid, gid, table)
            for gid, table in enumerate(tables)
        ])
    )
    event = (event_groups
        .loc[:, ['eid', 'place', 'date']]
        .drop_duplicates()
    )
    assert len(event.index) == 1
    groups = (event_groups
        .drop(columns=['place', 'date'])
        .merge(remarks, how='outer', validate='one_to_one')
    )
    return event, groups, standings, results

def _download_all(eids: Sequence[int]) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _download(eid)
            for eid in eids
        ])
    )

def _download_nats() -> pd.DataFrame:
    url = 'https://www.kleier.net/tournaments/byplace/index.php'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    return (pd
        .DataFrame(
            data=[
                (int(eid.get('href').split('=')[1]), nat.find('span').text)
                for nat in soup.find('ul', {'class': 'nat'}).find_all('li', recursive = False)
                for eid in nat.find_all('a')
            ],
            columns=['eid', 'nat']
        )
        .sort_values('eid')
        .reset_index(drop=True)
    )

def format_events(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ['eid', 'date', 'place', 'nat']]

def format_groups(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .loc[:, [
            'eid', 'gid',
            'name', 'group',
            'pW', 'pD', 'pL', 'M', 'N',
            'file_date', 'file_name', 'file_from', 'remarks'
        ]]
    )

def format_standings(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .pipe(lambda x: x
            .set_axis(x
                .columns
                .to_flat_index()
                .map('_'.join)
                , axis='columns', inplace=False
            )
        )
        .rename(columns=lambda x: x.strip('_'))
        .rename(columns=lambda x: re.sub(r'(.+)_\1', r'\1', x))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: x.replace('.', '_'))
        .rename(columns=lambda x: re.sub(r'(.*)name', r'\1', x))
        .rename(columns=lambda x: re.sub(r'rating_(.*)', r'\1', x))
        .rename(columns=lambda x: re.sub(r'standings_(.*)', r'\1', x))
        .rename(columns={
            '#'          : 'rank',
            'nationality': 'nat',
            'value'      : 'Rn',
            'change'     : 'dR'
        })
        .astype(dtype={column: int             for column in ['rank', 'score']})
        .astype(dtype={column: float           for column in ['Rn', 'dR']})
        .astype(dtype={column: pd.Int64Dtype() for column in ['Rn', 'dR']})
        .astype(dtype={column: float           for column in ['eff_games', 'buchholz', 'median']})
        .astype(dtype={'compa': 'category'})
        .assign(Ro = lambda x: x.Rn - x.dR)
        .loc[:, [
            'eid', 'gid', 'sur', 'pre', 'nat',
            'score', 'buchholz', 'median', 'compa',
            'rank', 
            'eff_games',
            'Ro', 'dR', 'Rn'
        ]]
    )

def format_results(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .rename(columns=lambda x: x.lower())
        .rename(columns={
            '##'     : 'rank1',
            'results': 'result'
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
        .loc[:, [
            'eid', 'gid', 'round', 'rank1', 'rank2',
            'result', 'unplayed'
        ]]
        .astype(dtype={column: int for column in ['rank1', 'rank2']})
        .astype(dtype={'result': 'category'})
    )

def main(max_eid: int) -> Tuple[pd.DataFrame]:
    events, groups, standings, results = _download_all(range(1, 1 + max_eid))
    assert events.equals(events.sort_values(['date', 'eid']))
    nats = _download_nats()
    events = pd.merge(events, nats, validate='one_to_one')
    kleier.utils._save_dataset(events, 'events')
    kleier.utils._save_dataset(groups, 'groups')
    kleier.utils._save_dataset(standings, 'standings')
    kleier.utils._save_dataset(results, 'results')
    return events, groups, standings, results

if __name__ == '__main__':
    sys.exit(main(int(sys.argv[1])))
