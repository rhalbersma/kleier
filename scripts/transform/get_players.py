#!/usr/bin/env python

#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import sys

import bs4
import numpy as np
import pandas as pd
import requests
from typing import Sequence, Tuple

import kleier.utils

def _parse_name_header(header: bs4.element.Tag) -> str:
    return header.text.split('History of ')[1]

def _parse_name_table(table: bs4.element.Tag) -> str:
    return table.attrs['summary'].split('Game Balance of ')[1]

def _player(pid: int, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        data   =[(pid,   name)],
        columns=['pid', 'name']
    )

def _read_games(pid: int, table: bs4.element.Tag) -> pd.DataFrame:
    return (pd
        .read_html(str(table), header=[1, 2])[0]
        .assign(
            pid = pid
        )
        .pipe(lambda x: x.loc[:, x.columns.to_list()[-1:] + x.columns.to_list()[:-1]])
        .assign(Unplayed = pd.Series([
            td['class'][0] == 'unplayed' if td.has_attr('class') else np.nan
            for tr in table.find_all('tr')[3:]
            for td in tr.find_all('td')[-1:]
        ]))
    )

def _get_player(pid: int) -> Tuple[pd.DataFrame]:
    assert 1 <= pid
    url = f'https://www.kleier.net/cgi/player.php?pid={pid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    header = soup.find('h1')
    table = soup.find('table')
    assert header or not table
    name = _parse_name_header(header) if header else np.nan
    if table:
        assert name == _parse_name_table(table)
    player = _player(pid, name)
    games = _read_games(pid, table) if table else None
    return player, games

def _get_all_players(pid_seq: Sequence[int]) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _get_player(pid)
            for pid in pid_seq
        ])
    )

def main(max_pid: int) -> Tuple[pd.DataFrame]:
    players, games = _get_all_players(range(1, 1 + max_pid))
    kleier.utils._save_dataset(players, 'players')
    kleier.utils._save_dataset(games, 'games')
    return players, games

if __name__ == '__main__':
    sys.exit(main(int(sys.argv[1])))
