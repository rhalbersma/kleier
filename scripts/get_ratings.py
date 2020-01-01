#!/usr/bin/env python

#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import sys

import bs4
import pandas as pd
import requests

import kleier.utils

def _get_rat_table(max_eid: int, max_pid: int) -> pd.DataFrame:
    # https://www.kleier.net/rating/general/index.php
    min, max = -9999, 9999
    from_, till = 'A', '['
    games = 0
    ntourn = max_eid
    items = max_eid * max_pid
    sortby = 'r'
    colsel, nat = 0, 'all'
    url = f'https://www.kleier.net/cgi/rat_table.php?min={min}&max={max}&from={from_}&till={till}&games={games}&ntourn={ntourn}&items={items}&sortby={sortby}&colsel={colsel}&nat[]={nat}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    table = soup.find('table')
    return pd.read_html(str(table))[0]

def main(max_eid: int, max_pid: int) -> pd.DataFrame:
    ratings = _get_rat_table(max_eid, max_pid)
    kleier.utils._save_dataset(ratings, 'ratings')
    return ratings

if __name__ == '__main__':
    sys.exit(main(int(sys.argv[1]), int(sys.argv[2])))
