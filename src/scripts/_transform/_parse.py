#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re
from typing import Sequence, Tuple

import bs4
import numpy as np
import pandas as pd

from scripts._extract import _scan
from scripts._extract import _soup

# helper functions for _player

def _name_from_header(header: bs4.element.Tag) -> str:
    return header.text.split('History of ')[1]

def _name_from_table(table: bs4.element.Tag) -> str:
    return table.attrs['summary'].split('Game Balance of ')[1]

def _name(pid: int, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        data   =[(pid,   name)],
        columns=['pid', 'name']
    )

def _expected(pid: int, table: bs4.element.Tag) -> pd.DataFrame:
    return (pd
        .read_html(str(table), header=[1, 2])[0]
        .assign(
            pid = pid
        )
        # make 'pid' the first column
        .pipe(lambda df: df
            .loc[:, df.columns.to_list()[-1:] + df.columns.to_list()[:-1]]
        )
        .assign(Unplayed = pd.Series([
            td['class'][0] == 'unplayed' if td.has_attr('class') else np.nan
            for tr in table.find_all('tr')[3:]
            for td in tr.find_all('td')[-1:]
        ]))
    )

# helper functions for _rat_table

def _long_rat_table(table: bs4.element.Tag) -> pd.DataFrame:
    rat_table = pd.read_html(str(table))[0]
    rating = rat_table.filter(regex='Rating').columns
    long_rat_table = (pd
        .melt(rat_table,
            id_vars=rat_table.drop(columns=rating).columns.tolist(),
            value_vars=rating.tolist(),
            value_name='Rating'
        )
    )
    long_rat_table.columns = pd.MultiIndex.from_tuples([
        column if isinstance(column, tuple) else tuple([column] * 4)
        for column in long_rat_table.columns.tolist()
    ])
    return long_rat_table

def _dates(long_rat_table: pd.DataFrame) -> pd.DataFrame:
    return (long_rat_table
        .filter(regex='variable')
        .drop_duplicates()
        .reset_index(drop=True)
    )

def _ratings(long_rat_table: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    return (long_rat_table
        .merge(dates
            .iloc[0:1, :]
            , how='right', validate='many_to_one'
        )
        .pipe(lambda df: df
            .drop(columns=df.filter(regex='variable'))
        )
    )

def _history(long_rat_table: pd.DataFrame) -> pd.DataFrame:
    return (long_rat_table
        .pipe(lambda df: df
            .drop(columns=df.filter(regex='Ranking|Games'))
        )
    )

# helper functions for _tourn_table

def _results_from(s: str) -> Tuple[str]:
    return re.split(r'(^.*)\s?(\d{4}-\d{2}-\d{2})\s(.*$)', s)[1:-1]

def _group(eid: int, gid: int, table_header_rows: bs4.element.ResultSet) -> pd.DataFrame:
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

def _cross_table(eid: int, gid: int, table: bs4.element.Tag) -> pd.DataFrame:
    return (pd
        .read_html(str(table), header=[2, 3])[0]
        .assign(
            eid = eid,
            gid = gid
        )
        # make 'eid' and 'gid' the first two columns
        .pipe(lambda df: df
            .loc[:, df.columns.to_list()[-2:] + df.columns.to_list()[:-2]]
        )
    )

def _unplayed_games(M: int, N: int, table: bs4.element.Tag) -> pd.DataFrame:
    return pd.DataFrame(
        data=[[
                not td.text or td.has_attr('class') and td['class'][0] == 'unplayed'
                for td in tr.find_all('td')[-N:]
            ]
            for tr in table.find_all('tr')[4:4+M]
        ],
        columns=pd.MultiIndex.from_tuples([
            ('Unplayed', str(n + 1))
            for n in range(N)
        ])
    )

def _activity(cross_table: pd.DataFrame) -> pd.DataFrame:
    return (cross_table
        .filter(regex='eid|Surname|Prename|Nationality|Rating')
    )

def _standings(cross_table: pd.DataFrame) -> pd.DataFrame:
    return (cross_table
        .filter(regex='eid|gid|#|Surname|Prename|Nationality|Standings')
    )

def _results(cross_table: pd.DataFrame) -> pd.DataFrame:
    return (pd
        .wide_to_long(cross_table
            .filter(regex='eid|gid|#|Results|Unplayed')
            .pipe(lambda df: df
                .set_axis(df
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

def _group_activity_standings_results(eid: int, gid: int, table: bs4.element.Tag) -> Tuple[pd.DataFrame]:
    cross_table = _cross_table(eid, gid, table)
    last_row = cross_table.tail(1)
    results_from = str(last_row.iloc[0, 2])
    sep = 'Results from: '
    if results_from.startswith(sep):
        cross_table.drop(last_row.index, inplace=True)
        file_from, file_date, file_name = _results_from(results_from.split(sep)[1])
    else:
        file_from, file_date, file_name = [ np.nan ] * 3
    # Elo (1978) notation:
    # M = number of players
    # N = number of games (here: number of rounds)
    M, N = cross_table.filter(regex='Results').shape
    cross_table = cross_table.join(_unplayed_games(M, N, table))
    group = (_group(eid, gid, table.find('thead').find_all('tr'))
        .assign(
            M = M,
            N = N,
            file_from = file_from,
            file_date = file_date,
            file_name = file_name
        )
    )
    activity = _activity(cross_table)
    standings = _standings(cross_table)
    results = _results(cross_table)
    return group, activity, standings, results

# main parsing API: _player, _rat_table, _tourn_table, _tournaments

def _player(pid: int, path: str) -> Tuple[pd.DataFrame]:
    soup = _soup._player(pid, path)
    header = soup.find('h1')
    table = soup.find('table')
    assert header or not table
    name_from_header = _name_from_header(header) if header else np.nan
    if table:
        assert name_from_header == _name_from_table(table)
    name = _name(pid, name_from_header)
    expected = _expected(pid, table) if table else None
    return name, expected

def _players(path: str) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _player(pid, path)
            for pid in _scan._files(r'player-\d+\.html', path)
        ])
    )

def _rat_table(path: str) -> Tuple[pd.DataFrame]:
    table = _soup._rat_table(path).find('table', {'summary': 'Stratego Rating'})
    long_rat_table = _long_rat_table(table)
    dates = _dates(long_rat_table)
    ratings = _ratings(long_rat_table, dates)
    history = _history(long_rat_table)
    return dates, ratings, history

def _tourn_table(eid: int, path: str) -> Tuple[pd.DataFrame]:
    soup = _soup._tourn_table(eid, path)
    cross_table_seq = soup.find_all('table', {'summary': 'Stratego Tournament Cross-Table'})
    remarks = pd.DataFrame(
        data=[
            (cross_table_seq.index(remarks.find_previous('table')), remarks.text)
            for remarks in soup.find_all('pre')
        ],
        columns=['gid', 'remarks']
    )
    groups, activity, standings, results = tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _group_activity_standings_results(eid, gid, cross_table)
            for gid, cross_table in enumerate(cross_table_seq)
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
    return event, groups, activity, standings, results

def _tourn_tables(path: str) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _tourn_table(eid, path)
            for eid in _scan._files(r'tourn_table-\d+\.html', path)
        ])
    )

def _tournaments(path: str) -> pd.DataFrame:
    return (pd
        .DataFrame(
            data=[
                (int(eid.get('href').split('=')[1]), nat.find('span').text)
                for nat in _soup._tournaments(path).find('ul', {'class': 'nat'}).find_all('li', recursive=False)
                for eid in nat.find_all('a')
            ],
            columns=['eid', 'nat']
        )
    )
