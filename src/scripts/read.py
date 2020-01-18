#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import bs4
import os

def _read(html_path: str, filename: str) -> bs4.BeautifulSoup:
    path = os.path.join(html_path, filename)
    assert os.path.exists(html_path) and filename.endswith('.html')
    return bs4.BeautifulSoup(open(path), 'lxml')

def _player(pid: int, html_path: str) -> bs4.BeautifulSoup:
    return _read(html_path, f'player-{pid}.html')

def _rat_table(html_path: str) -> bs4.BeautifulSoup:
    return _read(html_path, 'rat_table.html')

def _tourn_table(eid: int, html_path: str) -> bs4.BeautifulSoup:
    return _read(html_path, f'tourn_table-{eid}.html')

def _tournaments_byplace(html_path: str) -> bs4.BeautifulSoup:
    return _read(html_path, 'tournaments_byplace.html')
