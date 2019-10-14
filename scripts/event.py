#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import bs4
import pandas as pd
import requests

import kleier.utils

def _parse_table(eid, gid, table) -> tuple:
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

    event_index = pd.DataFrame(
        data   =[(eid,   gid,   name,   place,   date,   group,   W,   D,   L)],
        columns=['eid', 'gid', 'name', 'place', 'date', 'group', 'W', 'D', 'L']
    )

    df = pd.read_html(str(table), header=3)[0]
    last_row = df.tail(1)
    if str(last_row.iloc[0, 0]).startswith('Results from: '):
        df.drop(last_row.index, inplace=True)

    df.rename(columns=lambda x: str.lower(x).replace('.', '_'), inplace=True)
    df.rename(columns={
        '#'          : 'rank',
        'surname'    : 'sur',
        'prename'    : 'pre',
        'nationality': 'nat',
        'value'      : 'rating'
    }, inplace=True)
    df.rename(columns=lambda x: re.sub(r'(^\d+)', r'R\1', x), inplace=True)

    df['eid'] = eid
    df['gid'] = gid
    columns = df.columns.to_list()
    columns = columns[-2:] + columns[:-2]
    df = df[columns]

    df = df.astype(dtype={column: int             for column in ['rank', 'score']})
    df = df.astype(dtype={column: float           for column in ['rating', 'change']})
    df = df.astype(dtype={column: pd.Int64Dtype() for column in ['rating', 'change']})
    df = df.astype(dtype={column: float           for column in ['eff_games', 'buchholz', 'median']})
    rounds = list(df.filter(regex='R\d+'))
    event_table = df.drop(columns=rounds)

    df = pd.wide_to_long(df.filter(['eid', 'gid', 'rank'] + rounds), ['R'], i='rank', j='round')
    df.reset_index(inplace=True)
    df = df[['eid', 'gid', 'round', 'rank', 'R']]
    df.rename(columns={'rank': 'rank1'}, inplace=True)

    # Withdrawal == virtual loss (similar to how bye == virtual win).
    df['R'].fillna('0-', inplace=True)

    # Strip the players' color info from the scores.
    df['R'].replace(r'(\d+[+=-])[BW]', r'\1', inplace=True, regex=True)

    # Split the scores into an opponent's rank and the result.
    df['rank2']  = df['R'].apply(lambda x: x[:-1])
    df['result'] = df['R'].apply(lambda x: x[-1])
    df = df.astype(dtype={column: int for column in ['rank2']})
    event_cross = df.drop(columns='R')

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
            _parse_table(eid, gid, table)
            for gid, table in enumerate(tables)
        ])
    )

def download_all(eids=range(1, 660 + 1)):
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            download(eid)
            for eid in eids
        ])
    )

def main():
    event_index, event_table, event_cross = download_all()
    kleier.utils._save_dataset(event_index, 'event_index')
    kleier.utils._save_dataset(event_table, 'event_table')
    kleier.utils._save_dataset(event_cross, 'event_cross')
