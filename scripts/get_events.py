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

def _read_tourn_table(eid: int, gid: int, table: bs4.element.Tag) -> pd.DataFrame:
    return (pd
        .read_html(str(table), header=[2, 3])[0]
        .assign(
            eid = eid,
            gid = gid
        )
        .pipe(lambda x: x.loc[:, x.columns.to_list()[-2:] + x.columns.to_list()[:-2]])
    )

def _parse_group(eid: int, gid: int, table_header_rows: bs4.element.ResultSet) -> pd.DataFrame:
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
    score_W, score_D, score_L = [
        int(points)
        for points in scoring.split(': ')[1].split()
    ]
    return pd.DataFrame(
        data   =[(eid,   gid,   name,   place,   date,   group,   score_W,   score_D,   score_L)],
        columns=['eid', 'gid', 'name', 'place', 'date', 'group', 'score_W', 'score_D', 'score_L']
    )

def _parse_results_from(results_from: str) -> Tuple[str]:
    return re.split(r'(^.*)\s?(\d{4}-\d{2}-\d{2})\s(.*$)', results_from)[1:-1]

def _parse_unplayed_games(M: int, N: int, table: bs4.element.Tag) -> pd.DataFrame:
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
    return (pd
        .wide_to_long(tourn_table
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

def _parse_tourn_table(eid: int, gid: int, table: bs4.element.Tag) -> Tuple[pd.DataFrame]:
    tourn_table = _read_tourn_table(eid, gid, table)
    last_row = tourn_table.tail(1)
    results_from = str(last_row.iloc[0, 2])
    sep = 'Results from: '
    if results_from.startswith(sep):
        tourn_table.drop(last_row.index, inplace=True)
        file_from, file_date, file_name = _parse_results_from(results_from.split(sep)[1])
    else:
        file_from, file_date, file_name = [ np.nan ] * 3
    # Elo (1978) notation:
    # M = number of players
    # N = number of games (here: number of rounds)
    M, N = tourn_table.filter(regex='Results').shape
    tourn_table = tourn_table.join(_parse_unplayed_games(M, N, table))
    group = (_parse_group(eid, gid, table.find('thead').find_all('tr'))
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
    return group, standings, results

def _get_tourn_table(eid: int) -> Tuple[pd.DataFrame]:
    assert 1 <= eid
    url = f'https://www.kleier.net/cgi/tourn_table.php?eid={eid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    table_seq = soup.find_all('table', {'summary': 'Stratego Tournament Cross-Table'})
    remarks = pd.DataFrame(
        data=[
            (table_seq.index(remarks.find_previous('table')), remarks.text)
            for remarks in soup.find_all('pre')
        ],
        columns=['gid', 'remarks']
    )
    groups, standings, results = tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _parse_tourn_table(eid, gid, table)
            for gid, table in enumerate(table_seq)
        ])
    )
    event = (groups
        .loc[:, ['eid', 'place', 'date']]
        .drop_duplicates()
    )
    assert len(event.index) == 1
    groups = (groups
        .drop(columns=['place', 'date'])
        .merge(remarks, how='outer', validate='one_to_one')
    )
    return event, groups, standings, results

def _get_all_tourn_tables(eid_seq: Sequence[int]) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _get_tourn_table(eid)
            for eid in eid_seq
        ])
    )

def _get_tournaments_byplace() -> pd.DataFrame:
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

def main(max_eid: int) -> Tuple[pd.DataFrame]:
    events, groups, standings, results = _get_all_tourn_tables(range(1, 1 + max_eid))
    assert events.equals(events.sort_values(['date', 'eid']))
    countries = _get_tournaments_byplace()
    events = pd.merge(events, countries, validate='one_to_one')
    kleier.utils._save_dataset(events, 'events')
    kleier.utils._save_dataset(groups, 'groups')
    kleier.utils._save_dataset(standings, 'standings')
    kleier.utils._save_dataset(results, 'results')
    return events, groups, standings, results

if __name__ == '__main__':
    sys.exit(main(int(sys.argv[1])))
