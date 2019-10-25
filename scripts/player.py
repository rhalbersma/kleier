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

def download(pid) -> tuple:
    url = f'https://www.kleier.net/cgi/player.php?pid={pid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    header = soup.find_all('h1')
    table  = soup.find_all('table')
    assert header or not table
    name = '' if not header else header[0].text.split('History of ')[1]
    player_index = pd.DataFrame(
        data   =[(pid,   name)],
        columns=['pid', 'name']
    )
    player_cross = None if not table else (pd
        .read_html(str(table[0]), header=2)[0]
        .assign(pid = pid)
    )
    return player_index, player_cross

def download_all(pids) -> tuple:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            download(pid)
            for pid in pids
        ])
    )

def parse_cross(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns.to_list()
    return (df
        .loc[:, columns[-1:] + columns[:-1]]
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: x.replace(' ', '_'))
        .rename(columns=lambda x: re.sub(r'(date).*', r'\1', x))
        .rename(columns={
            'pid'    : 'pid1',
            'surname': 'sur2',
            'prename': 'pre2',
            'rating' : 'rating2'
        })
        .astype(dtype={'date'   : 'datetime64[ns]'})
        .astype(dtype={'rating2': pd.Int64Dtype()})
    )

def main():
    player_index, player_cross = download_all(range(1, 1 + 2954))
    kleier.utils._save_dataset(player_index, 'player_index')
    kleier.utils._save_dataset(player_cross, 'player_cross')

if __name__ == '__main__':
    sys.exit(main())
