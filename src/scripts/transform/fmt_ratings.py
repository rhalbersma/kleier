#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import numpy as np
import pandas as pd
from typing import Tuple

def _lists(rat_table: pd.DataFrame) -> pd.DataFrame:
    df = (rat_table
        .filter(regex='Rating')
        .columns
        .to_frame()
        .reset_index(drop=True)
        .reset_index()
        .drop(columns=[0])
        .rename(columns={
            'index': 'id',
            1: 'place',
            2: 'date',
            3: 'significance'
        })
        .astype(dtype={
            'date'        : 'datetime64[ns]',
            'significance': float
        })
        .loc[:, [
            'id', 'date', 'place', 'significance'
        ]]
    )
    assert df.equals(df.sort_values('id'))
    return df

def _long_rat_table(rat_table: pd.DataFrame, lists: pd.DataFrame) -> pd.DataFrame:
    return (pd
        .wide_to_long(rat_table
            .pipe(lambda x: x
                .set_axis(pd.
                    Index(
                        [ t[2:] for t in x.drop(x.filter(regex='Rating'), axis='columns').columns ] +
                        [ ('Rating', str(id)) for id in lists['id'] ]
                    )
                    .to_flat_index()
                    .map(dict.fromkeys)
                    .map(list)
                    .map('_'.join)
                    , axis='columns', inplace=False
                )
            )
            .reset_index()
            , ['Rating_'], i='index', j='list_id'
        )
        .reset_index()
        .merge(lists
            .loc[:, ['id', 'date', 'place']]
            .rename(columns={'id': 'list_id'})
            , how='left', on='list_id', validate='many_to_one'
        )
        .rename(columns=lambda x: x.strip('._'))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: re.sub(r'ranking(_)(\w{3}).*', r'\2\1rank', x))
        .rename(columns=lambda x: re.sub(r'(games)(_)(\w{3}).*', r'\3\2\1', x))
        .rename(columns=lambda x: re.sub(r'(.*)name', r'\1', x))
        .rename(columns={'rating': 'R'})
        .replace({'R': {'unrated': np.nan}})
        .assign(nat = lambda x: x.nat_rank.str.split('/').str[-1])
        .assign(nat_rank = lambda x:
            np.where(
                x.nat_rank.str[0].str.isdigit(),
                x.nat_rank.str.split('/').str[0],
                np.nan
            )
        )
        .assign(int_rank = lambda x:
            np.where(
                x.int_rank.str[0].str.isdigit(),
                x.int_rank,
                np.nan
            )
        )
        .astype(dtype={column: float           for column in ['R', 'int_rank', 'nat_rank']})
        .astype(dtype={column: pd.Int64Dtype() for column in ['R', 'int_rank', 'nat_rank']})
        .loc[:, [
            'list_id', 'date', 'place', 'pre', 'sur', 'nat',
            'R', 'int_rank', 'nat_rank', 'eff_games', 'tot_games'
        ]]
    )

def format_ratings(df: pd.DataFrame) -> Tuple[pd.DataFrame]:
    lists = _lists(df)
    long_rat_table = _long_rat_table(df, lists)
    ratings = (long_rat_table
        .query('list_id == 0')
        .drop(columns=['list_id', 'date', 'place'])
    )
    history = (long_rat_table
        .drop(columns=['int_rank', 'nat_rank', 'eff_games', 'tot_games'])
        .query('R.notnull()')
        .reset_index()
        .sort_values(
            ['list_id', 'R', 'index'],
            ascending=[False, False, True]
        )
        .drop(columns=['index', 'list_id'])
        .reset_index(drop=True)
    )
    lists = (lists
        .sort_values('id', ascending=False)
        .reset_index(drop=True)
        .drop(columns=['id'])
    )
    return ratings, history, lists
