#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import bs4
import pandas as pd
import requests

import kleier.utils

def parse_table(eid, gid, table) -> tuple:
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
        int(score)
        for score in scoring.split(': ')[1].split()
    ]

    events = pd.DataFrame(
        data=[(eid, gid, name, place, date, group, W, D, L)],
        columns=['eid', 'gid', 'name', 'place', 'date', 'group', 'W', 'D', 'L']
    )

    df = pd.read_html(str(table), header=3)[0].head(-1)
    df.rename(columns={'#': 'rank', 'Surname': 'sur', 'Prename': 'pre'}, inplace=True)
    df.rename(columns=lambda x: re.sub(r'(^\d+)', r'R\1', x), inplace=True)

    # Mark missing names as "Anonymous".
    df.fillna({column: 'Anonymous' for column in ['sur', 'pre']}, inplace=True)

    df['eid'] = eid
    df['gid'] = gid
    columns = df.columns.to_list()
    columns = columns[-2:] + columns[:-2]
    df = df[columns]

    df = df.astype(dtype={column: int             for column in ['rank', 'Score']})
    df = df.astype(dtype={column: float           for column in ['Value', 'Change']})
    df = df.astype(dtype={column: pd.Int64Dtype() for column in ['Value', 'Change']})
    df = df.astype(dtype={column: float           for column in ['Eff.Games', 'Buchholz', 'Median']})
    rounds = list(df.filter(regex='R\d+'))
    standings = df.drop(columns=rounds)

    df = pd.wide_to_long(df.filter(['eid', 'gid', 'rank'] + rounds), ['R'], i='rank', j='round')
    df.reset_index(inplace=True)
    columns = df.columns.to_list()
    columns = columns[1:-1][::-1] + [columns[0]] + [columns[-1]]
    df = df[columns]

    # Withdrawal == virtual loss (similar to how bye == virtual win).
    df['R'].fillna('0-', inplace=True)

    # Strip the players' color info from the scores.
    df['R'].replace(r'(\d+[+=-])[BW]', r'\1', inplace=True, regex=True)

    # Split the scores into an opponent's rank and the result.
    df['opponent'] = df['R'].apply(lambda x: x[:-1])
    df['result']   = df['R'].apply(lambda x: x[-1])
    df = df.astype(dtype={column: int for column in ['opponent']})
    results = df.drop(columns='R')

    return events, standings, results

def tourn_table(eid) -> tuple:
    url = f'https://www.kleier.net/cgi/tourn_table.php?eid={eid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    tables = soup.find_all('table', {'summary': 'Stratego Tournament Cross-Table'})
    return tuple(
        pd.concat(list(t))
        for t in zip(*[
            parse_table(eid, gid, table)
            for gid, table in enumerate(tables)
        ])
    )

def main():
    events, standings, results = tuple(
        pd.concat(list(t))
        for t in zip(*[
            tourn_table(eid)
            for eid in range(1, 661)
        ])
    )
    kleier.utils.save_dataset(events, 'events')
    kleier.utils.save_dataset(standings, 'standings')
    kleier.utils.save_dataset(results, 'results')
