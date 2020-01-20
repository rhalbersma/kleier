#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import os

import bs4

def _read(path: str, file: str) -> bs4.BeautifulSoup:
    assert os.path.exists(path) and file.endswith('.html')
    return bs4.BeautifulSoup(open(os.path.join(path, file)), 'lxml')

def _player(pid: int, path: str) -> bs4.BeautifulSoup:
    return _read(path, f'player-{pid}.html')

def _rat_table(path: str) -> bs4.BeautifulSoup:
    return _read(path, 'rat_table.html')

def _tourn_table(eid: int, path: str) -> bs4.BeautifulSoup:
    return _read(path, f'tourn_table-{eid}.html')

def _tournaments_byplace(path: str) -> bs4.BeautifulSoup:
    return _read(path, 'tournaments_byplace.html')
