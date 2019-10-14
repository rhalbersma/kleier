#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

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
    tables = soup.find_all('table')
    assert header or not tables
    name = '' if not header else header[0].text.split('History of ')[1]
    player_index = pd.DataFrame(
        data   =[(pid,   name)],
        columns=['pid', 'name']
    )
    player_cross = None if not tables else (pd
        .read_html(str(tables[0]), header=2)[0]
        .assign(
            pid1  = pid,
            name1 = name
        )
    )
    return player_index, player_cross

def download_all(pids=range(1, 2952 + 1)) -> tuple:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            download(pid)
            for pid in pids
        ])
    )

def parse(player_cross: pd.DataFrame) -> pd.DataFrame:
    columns = player_cross.columns.to_list()
    return (player_cross
        .loc[:, columns[-2:] + columns[:-2]]
        .rename(columns=lambda x: str.lower(x).replace(' ', '_'))
        .rename(columns=lambda x: 'date' if str(x).startswith('date') else x)
        .rename(columns={
            'surname': 'sur2',
            'prename': 'pre2',
            'rating' : 'rating2'
        })
        .astype(dtype={'date'   : 'datetime64[ns]'})
        .astype(dtype={'rating2': pd.Int64Dtype()})
        .sort_values(
            by       =['pid1', 'date', 'rating2'],
            ascending=[ True,   True,   False   ]
        )
        .reset_index(drop=True)
    )

def main():
    player_index, player_cross = download_all()
    kleier.utils._save_dataset(player_index, 'player_index')
    kleier.utils._save_dataset(player_cross, 'player_cross')
