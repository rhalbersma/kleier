#!/usr/bin/env python

#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import math
import re
import sys

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
    ntourn = max_eid
    items = max_pid * max_eid
    sortby = 'r'
    colsel, nat = 0, 'all'

    url = f'https://www.kleier.net/cgi/rat_table.php?min={min}&max={max}&from={from_}&till={till}&games={games}&ntourn={ntourn}&items={items}&sortby={sortby}&colsel={colsel}&nat[]={nat}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    table = soup.find('table')
    return pd.read_html(str(table))[0]

def _significance(dates: pd.Series) -> pd.Series:
    max_dates = dates.max()
    decay, tropical_year = 2.5731, 365.246
    return np.round(np.exp(-((max_dates - dates).dt.days / (decay * tropical_year))**2), 6)

def format_ratings(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rating_lists = (df
        .filter(regex='Rating')
        .columns
        .to_frame()
        .reset_index(drop=True)
        .drop(columns=[0])
        .rename(columns={
            1: 'place',
            2: 'date',
            3: 'significance'
        })
        .astype(dtype={
            'date'        : 'datetime64[ns]',
            'significance': float
        })
        .merge(events, validate='one_to_one')
        .loc[:, ['eid', 'date', 'place', 'nat', 'significance']]
    )
    assert rating_lists.equals(rating_lists.sort_values('eid', ascending=False))
    assert np.isclose(_significance(rating_lists['date']), rating_lists['significance'], rtol=1e-4).all()
    return (df
        .pipe(lambda x: x
            .set_axis(pd.
                Index(
                    [ t[2:] for t in x.drop(x.filter(regex='Rating'), axis='columns').columns ] +
                    [ ('R', str(eid)) for eid in rating_lists['eid'] ]
                )
                .to_flat_index()
                .map(dict.fromkeys)
                .map(list)
                .map('_'.join)
                , axis='columns', inplace=False
            )
        )
        .rename(columns=lambda x: x.strip('.'))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: re.sub(r'ranking(_)(\w{3}).*', r'\2\1rank', x))
        .rename(columns=lambda x: re.sub(r'(games)(_)(\w{3}).*', r'\3\2\1', x))
        .rename(columns=lambda x: re.sub(r'(.*)name', r'\1', x))
        .rename(columns=lambda x: re.sub(r'r_(\d+)', r'R\1', x))
        .assign(nat = lambda x: x.nat_rank.str.split('/').str[-1])
        .assign(int_rank = lambda x:
            np.where(
                x.int_rank.str[0].str.isdigit(),
                x.int_rank,
                np.nan
            )
        )
        .assign(nat_rank = lambda x:
            np.where(
                x.nat_rank.str[0].str.isdigit(),
                x.nat_rank.str.split('/').str[0],
                np.nan
            )
        )
        .replace({'unrated': np.nan})
        .pipe(lambda x: x.astype(dtype={column: float           for column in x.columns if column.endswith('rank')}))
        .pipe(lambda x: x.astype(dtype={column: pd.Int64Dtype() for column in x.columns if column.endswith('rank')}))
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
