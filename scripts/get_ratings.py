#!/usr/bin/env python

#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import bs4
import numpy as np
import pandas as pd
import requests

import kleier.utils

def _download(max_eid: int, max_pid: int) -> pd.DataFrame:
    # https://www.kleier.net/rating/general/index.php
    min, max = -9999, 9999
    from_, till = 'A', '['
    games = 0
    tourn = max_eid
    items = max_pid * max_eid
    sortby = 'r'
    colsel, nat = 0, 'all'

    url = f'https://www.kleier.net/cgi/rat_table.php?min={min}&max={max}&from={from_}&till={till}&games={games}&ntourn={tourn}&items={items}&sortby={sortby}&colsel={colsel}&nat[]={nat}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    table = soup.find('table')
    return pd.read_html(str(table))[0]

def format_ratings(df: pd.DataFrame, max_eid: int) -> pd.DataFrame:
    tourn = df.shape[1] - 6
    return (df
        .pipe(lambda x:
            x.set_axis(x
                .columns
                .to_flat_index()
                .map(dict.fromkeys)
                .map(list)
                .map('_'.join)
                , axis='columns', inplace=False
            )
        )
        .rename(columns=lambda x: x.strip('.'))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: re.sub(r'(games)(_)(.{3}).*', r'\3\2\1', x))
        .rename(columns=lambda x: re.sub(r'(.*)name', r'\1', x))
        .pipe(lambda x: x.rename(columns = {
                x.columns[6 + (max_eid - eid)] : 'R' + str(eid)
                for eid in reversed(range(1 + max_eid - tourn, 1 + max_eid))
            })
        )
        .assign(pnat = lambda x: x.ranking_natl.str.split('/').str[-1])
        .assign(ranking_intl = lambda x:
            np.where(
                x.ranking_intl.str[0].str.isdigit(),
                x.ranking_intl,
                None
            )
        )
        .assign(ranking_natl = lambda x:
            np.where(
                x.ranking_natl.str[0].str.isdigit(),
                x.ranking_natl.str.split('/').str[0],
                None
            )
        )
        .replace({'unrated': None})
        .pipe(lambda x: x.astype(dtype={column: float           for column in x.columns if column.startswith('ranking')}))
        .pipe(lambda x: x.astype(dtype={column: pd.Int64Dtype() for column in x.columns if column.startswith('ranking')}))
        .pipe(lambda x: x.astype(dtype={column: float           for column in x.columns if column.startswith('R')}))
        .pipe(lambda x: x.astype(dtype={column: pd.Int64Dtype() for column in x.columns if column.startswith('R')}))
        .pipe(lambda x: x.loc[:, x.columns.to_list()[:6] + x.columns.to_list()[-1:] + x.columns.to_list()[6:-1]])
    )

def main(max_eid: int, max_pid: int) -> pd.DataFrame:
    ratings = _download(max_eid, max_pid)
    kleier.utils._save_dataset(ratings, 'ratings')
    return ratings

if __name__ == '__main__':
    sys.exit(main(int(sys.argv[1]), int(sys.argv[2])))
