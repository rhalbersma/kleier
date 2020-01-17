#!/usr/bin/env python

#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import os
import sys

import requests

def _wget(prefix: str, file: str, *urls) -> None:
    """
    wget -P prefix -O file urls
    """
    os.makedirs(prefix, exist_ok=True)
    for url in urls:
        response = requests.get(url)
        assert response.status_code == 200
        with open(os.path.join(prefix, file), 'wb') as dst:
            dst.write(response.content)

def tourn_table(prefix: str, eid: int) -> None:
    file = f'tourn_table-{eid}.html'
    url = f'https://www.kleier.net/cgi/tourn_table.php?eid={eid}'
    _wget(prefix, file, url)

def player(prefix: str, pid: int) -> None:
    file = f'player-{pid}.html'
    url = f'https://www.kleier.net/cgi/player.php?pid={pid}'
    _wget(prefix, file, url)

def tournaments_byplace(prefix: str) -> None:
    file = 'tournaments_byplace.html'
    url = 'https://www.kleier.net/tournaments/byplace/index.php'
    _wget(prefix, file, url)

def rat_table(prefix: str, min=-9999, max=9999, from_='A', till='[', games=1, ntourn=12, items=2500, sortby='r', colsel=0, nat='all') -> None:
    file = 'rat_table.html'
    url = f'https://www.kleier.net/cgi/rat_table.php?min={min}&max={max}&from={from_}&till={till}&games={games}&ntourn={ntourn}&items={items}&sortby={sortby}&colsel={colsel}&nat[]={nat}'
    _wget(prefix, file, url)

def main(prefix: str, max_eid: int, max_pid: int) -> None:
    for eid in range(1, 1 + max_eid):
        tourn_table(prefix, eid)
    tournaments_byplace(prefix)
    for pid in range(1, 1 + max_pid):
        player(prefix, pid)
    rat_table(prefix, ntourn=max_eid, items=max_eid*max_pid)

if __name__ == '__main__':
    sys.exit(main(str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])))
