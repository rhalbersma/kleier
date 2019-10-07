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
    pts_W, pts_D, pts_L = [
        int(points)
        for points in scoring.split(': ')[1].split()
    ]

    event_info = pd.DataFrame(
        data   =[(eid,   gid,   name,   place,   date,   group,   pts_W,   pts_D,   pts_L)],
        columns=['eid', 'gid', 'name', 'place', 'date', 'group', 'pts_W', 'pts_D', 'pts_L']
    )

    df = pd.read_html(str(table), header=3)[0]
    last_row = df.tail(1)
    if str(last_row.iloc[0, 0]).startswith("Results from:"):
        df.drop(last_row.index, inplace=True)

    df.rename(columns=lambda x: str.lower(x).replace('.', '_'), inplace=True)
    df.rename(columns={
        '#'      : 'rank', 
        'surname': 'sur', 
        'prename': 'pre',
        'value'  : 'rating'
    }, inplace=True)
    df.rename(columns=lambda x: re.sub(r'(^\d+)', r'r\1', x), inplace=True)

    # Mark missing names as "Anonymous".
    df.fillna({column: 'Anonymous' for column in ['sur', 'pre']}, inplace=True)

    df['eid'] = eid
    df['gid'] = gid
    columns = df.columns.to_list()
    columns = columns[-2:] + columns[:-2]
    df = df[columns]

    df = df.astype(dtype={column: int             for column in ['rank', 'score']})
    df = df.astype(dtype={column: float           for column in ['rating', 'change']})
    df = df.astype(dtype={column: pd.Int64Dtype() for column in ['rating', 'change']})
    df = df.astype(dtype={column: float           for column in ['eff_games', 'buchholz', 'median']})
    rounds = list(df.filter(regex='r\d+'))
    standings = df.drop(columns=rounds)

    df = pd.wide_to_long(df.filter(['eid', 'gid', 'rank'] + rounds), ['r'], i='rank', j='round')
    df.reset_index(inplace=True)
    columns = df.columns.to_list()
    columns = columns[2:4][::-1] + columns[0:2][::-1] + [columns[-1]]
    df = df[columns]
    df.rename(columns={'rank': 'rank1'}, inplace=True)

    # Withdrawal == virtual loss (similar to how bye == virtual win).
    df['r'].fillna('0-', inplace=True)

    # Strip the players' color info from the scores.
    df['r'].replace(r'(\d+[+=-])[BW]', r'\1', inplace=True, regex=True)

    # Split the scores into an opponent's rank and the result.
    df['rank2']  = df['r'].apply(lambda x: x[:-1])
    df['result'] = df['r'].apply(lambda x: x[-1])
    df = df.astype(dtype={column: int for column in ['rank2']})
    cross_table = df.drop(columns='r')

    return event_info, standings, cross_table

def _tourn_table(eid) -> tuple:
    url = f'https://www.kleier.net/cgi/tourn_table.php?eid={eid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    tables = soup.find_all('table', {'summary': 'Stratego Tournament Cross-Table'})
    return tuple(
        pd.concat(list(t), sort=False).reset_index(drop=True)
        for t in zip(*[
            _parse_table(eid, gid, table)
            for gid, table in enumerate(tables)
        ])
    )

def download(num_eid=660):
    event_info, standings, cross_table = tuple(
        pd.concat(list(t), sort=False).reset_index(drop=True)
        for t in zip(*[
            _tourn_table(eid)
            for eid in range(1, num_eid + 1)
        ])
    )
    kleier.utils._save_dataset(event_info, 'event_info')
    kleier.utils._save_dataset(standings, 'standings')
    kleier.utils._save_dataset(cross_table, 'cross_table')