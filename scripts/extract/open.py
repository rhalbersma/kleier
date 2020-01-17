#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import bs4
import os

def _open(prefix: str, file: str) -> bs4.BeautifulSoup:
    path = os.path.join(prefix, file)
    assert os.path.exists(path) and file.endswith('.html')
    return bs4.BeautifulSoup(open(os.path.join(prefix, file)), 'lxml')

def tourn_table(prefix: str, eid: int) -> bs4.BeautifulSoup:
    file = f'tourn_table-{eid}.html'
    return _open(prefix, file)

def tournaments_byplace(prefix: str) -> bs4.BeautifulSoup:
    file = 'tournaments_byplace.html'
    return _open(prefix, file)

def player(prefix: str, pid: int) -> bs4.BeautifulSoup:
    file = f'player-{pid}.html'
    return _open(prefix, file)

def rat_table(prefix: str) -> bs4.BeautifulSoup:
    file = 'rat_table.html'
    return _open(prefix, file)
