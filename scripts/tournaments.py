#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import bs4
import pandas as pd
import requests

import kleier.utils

def parse_epd(table) -> tuple:
    event_place_date = table.find('thead').find_all('tr')[0].find('th').text
    split = event_place_date.split('\xa0\xa0\xa0\xa0\xa0\xa0')
    event = '' if len(split) == 1 else split[0]
    place_date = split[0] if len(split) == 1 else split[1]
    split = place_date.split()
    place = ' '.join(split[:-1])
    date = pd.to_datetime(split[-1])
    return event, place, date

def parse_gs(table) -> tuple:
    group_scoring = table.find('thead').find_all('tr')[1].find('th').text
    group, scoring = group_scoring.split('\xa0\xa0')
    group = group.split(': ')[1][:-1]
    W, D, L = [ 
        int(score) 
        for score in scoring.split(': ')[1].split() 
    ]
    return group, W, D, L

def parse_sr(table, eid, gid) -> (pd.DataFrame, pd.DataFrame):
    df = pd.read_html(str(table), header=3)[0].head(-1)
    df.rename(columns={'#': 'rank', 'Surname': 'sur', 'Prename': 'pre'}, inplace=True)
    df.rename(columns=lambda x: re.sub(r'(^\d+)', r'R\1', x), inplace=True)    
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
    df['R'] = df['R'].str.replace(r'(\d+[+=-])[BW]', r'\1', regex=True)
    df['opponent'] = df['R'].apply(lambda x: x if pd.isna(x) else x[:-1])
    df = df.astype(dtype={column: float           for column in ['opponent']})
    df = df.astype(dtype={column: pd.Int64Dtype() for column in ['opponent']})
    df['result'] = df['R'].apply(lambda x: x if pd.isna(x) else x[-1])
    results = df.drop(columns='R')

    return standings, results

def tourn_table(eid) -> tuple:
    url = f'https://www.kleier.net/cgi/tourn_table.php?eid={eid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    tables = soup.find_all('table', {'summary': 'Stratego Tournament Cross-Table'})
    event_place_date = pd.DataFrame(
        data=[
            (eid, gid) + parse_epd(table) + parse_gs(table)
            for gid, table in enumerate(tables)
        ],
        columns=['eid', 'gid', 'event', 'place', 'date', 'group', 'W', 'D', 'L']
    )
    standings_results = [
        parse_sr(table, eid, idx)
        for idx, table in enumerate(tables)
    ]
    standings = pd.concat([
        sr[0]
        for sr in standings_results
    ])
    results = pd.concat([
        sr[1]
        for sr in standings_results
    ])
    return event_place_date, standings, results

def main():
    tourn_tables = [
        tourn_table(eid)
        for eid in range(1, 661)
    ]
    events = pd.concat([
        tourn_table[0]
        for tourn_table in tourn_tables
    ])
    standings = pd.concat([
        tourn_table[1]
        for tourn_table in tourn_tables
    ])
    results = pd.concat([
        tourn_table[2]
        for tourn_table in tourn_tables
    ])
    kleier.utils.save_dataset(events, 'events')
    kleier.utils.save_dataset(standings, 'standings')
    kleier.utils.save_dataset(results, 'results')
