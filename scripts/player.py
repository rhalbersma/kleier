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

def _header_name(header: bs4.element.Tag) -> str:
    return header.text.split('History of ')[1]

def _table_name(table: bs4.element.Tag) -> str:
    return table.attrs['summary'].split('Game Balance of ')[1]

def _player(pid: int, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        data   =[(pid,   name)],
        columns=['pid', 'name']
    )

def _games(pid: int, table: bs4.element.Tag) -> pd.DataFrame:
    return (pd
        .read_html(str(table), header=[1, 2])[0]
        .assign(pid = pid)
    )

def download(pid: int) -> Tuple[pd.DataFrame]:
    url = f'https://www.kleier.net/cgi/player.php?pid={pid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    header = soup.find('h1')
    table = soup.find('table')
    assert header or not table
    name = '' if not header else _header_name(header)
    if table:
        assert name == _table_name(table)
    player = _player(pid, name)
    games = None if not table else _games(pid, table)
    return player, games

def download_all(pids: Sequence[int]) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            download(pid)
            for pid in pids
        ])
    )

def parse_game_balance(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [ ' '.join(v) for v in df.columns.values ]
    columns = df.columns.to_list()
    return (df
        .loc[:, columns[-1:] + columns[:-1]]
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: x.replace(' ', '_'))
        .rename(columns=lambda x: re.sub(r'(date).+', r'\1', x))
        .rename(columns=lambda x: re.sub(r'event_(.+)', r'\1', x))
        .rename(columns=lambda x: re.sub(r'opponent_(.+)', r'\g<1>2', x))
        .rename(columns=lambda x: re.sub(r'result_(.+)', r'\1', x))
        .rename(columns=lambda x: re.sub(r'(.+)name(2)', r'\1\2', x))
        .astype(dtype={'date'   : 'datetime64[ns]'})
        .astype(dtype={'rating2': pd.Int64Dtype()})
    )

def main():
    players, games = download_all(range(1, 1 + 2954))
    kleier.utils._save_dataset(players, 'players')
    kleier.utils._save_dataset(games, 'games')

if __name__ == '__main__':
    sys.exit(main())
