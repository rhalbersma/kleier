#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import os

import requests

def _wget(directory_prefix: str, output_document: str, *urls) -> None:
    os.makedirs(directory_prefix, exist_ok=True)
    with open(os.path.join(directory_prefix, output_document), 'wb') as dst:
        for url in urls:
            response = requests.get(url)
            assert response.status_code == 200
            dst.write(response.content)

def _player(pid: int, path: str) -> None:
    filename = f'player-{pid}.html'
    url = f'https://www.kleier.net/cgi/player.php?pid={pid}'
    _wget(path, filename, url)

def _rat_table(path: str, min=-9999, max=9999, from_='A', till='[', games=1, ntourn=12, items=2500, sortby='r', colsel=0, nat='all') -> None:
    filename = 'rat_table.html'
    url = f'https://www.kleier.net/cgi/rat_table.php?min={min}&max={max}&from={from_}&till={till}&games={games}&ntourn={ntourn}&items={items}&sortby={sortby}&colsel={colsel}&nat[]={nat}'
    _wget(path, filename, url)

def _tourn_table(eid: int, path:str) -> None:
    filename = f'tourn_table-{eid}.html'
    url = f'https://www.kleier.net/cgi/tourn_table.php?eid={eid}'
    _wget(path, filename, url)

def _tournaments_byplace(path: str) -> None:
    filename = 'tournaments_byplace.html'
    url = 'https://www.kleier.net/tournaments/byplace/index.php'
    _wget(path, filename, url)
