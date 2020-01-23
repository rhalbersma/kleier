#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)


import re
from typing import Tuple

import numpy as np
import pandas as pd

def _tournaments_byplace(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .sort_values('eid')
        .reset_index(drop=True)
    )

def _events(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .loc[:, [
            'eid', 'date', 'place'
        ]]
        .sort_values('eid')
        .reset_index(drop=True)
        .rename(columns={'eid' : 'id'})
    )

def _groups(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .sort_values(['eid', 'gid'])
        .reset_index()
        .drop(columns=['gid'])
        .rename(columns={
            'index': 'id',
            'eid'  : 'event_id'
        })
    )

def _activity(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .pipe(lambda x: x
            .set_axis(x
                .columns
                .to_flat_index()
                .map('_'.join)
                , axis='columns', inplace=False
            )
        )
        .rename(columns=lambda x: x.strip('_'))
        .rename(columns=lambda x: re.sub(r'(.+)_\1', r'\1', x))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: x.replace('.', '_'))
        .rename(columns=lambda x: re.sub(r'(.*)name', r'\1', x))
        .rename(columns=lambda x: re.sub(r'rating_(.*)', r'\1', x))
        .rename(columns={
            'eid'        : 'event_id',
            'nationality': 'nat',
            'value'      : 'Rn',            # Rn = a player's rating after a performance (Elo, 1978)
            'change'     : 'dR'             # dR = Rn - Ro ('d' from 'delta')
        })
        .astype(dtype={column: float           for column in ['Rn', 'dR']})
        .astype(dtype={column: pd.Int64Dtype() for column in ['Rn', 'dR']})
        .astype(dtype={column: float           for column in ['eff_games']})
        .assign(Ro = lambda x: x.Rn - x.dR) # Ro = a player's rating before a performance (Elo, 1978)
        .loc[:, [
            'pre', 'sur', 'nat', 'event_id',
            'eff_games', 'Rn', 'Ro', 'dR'
        ]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

def _standings(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .pipe(lambda x: x
            .set_axis(x
                .columns
                .to_flat_index()
                .map('_'.join)
                , axis='columns', inplace=False
            )
        )
        .rename(columns=lambda x: x.strip('_'))
        .rename(columns=lambda x: re.sub(r'(.+)_\1', r'\1', x))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: x.replace('.', '_'))
        .rename(columns=lambda x: re.sub(r'(.*)name', r'\1', x))
        .rename(columns=lambda x: re.sub(r'rating_(.*)', r'\1', x))
        .rename(columns=lambda x: re.sub(r'standings_(.*)', r'\1', x))
        .rename(columns={
            'eid'        : 'event_id',
            '#'          : 'rank',
            'nationality': 'nat'
        })
        .assign(
            dmr_W = lambda x: np.where(x.compa.isnull(), 0, x.compa.str.split('/').str[0]),
            dmr_N = lambda x: np.where(x.compa.isnull(), 0, x.compa.str.split('/').str[1])
        )
        .astype(dtype={column: int   for column in ['rank', 'score', 'dmr_W', 'dmr_N']})
        .astype(dtype={column: float for column in ['median', 'buchholz']})
        .pipe(lambda x: x
            .merge(x
                .loc[:, ['event_id', 'gid']]
                .drop_duplicates(subset=['event_id', 'gid'])
                .sort_values(['event_id', 'gid'])
                .reset_index(drop=True)
                .reset_index()
                .rename(columns={'index': 'group_id'})
            )
        )
        .loc[:, [
            'group_id', 'pre', 'sur', 'nat',
            'rank', 'score', 'median', 'buchholz', 'dmr_W', 'dmr_N'
        ]]
    )

def _results(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .rename(columns=lambda x: x.lower())
        .rename(columns={
            'eid'    : 'event_id',
            '##'     : 'rank_1',
            'results': 'result'
        })
        .fillna({'result': '0?'})
        .replace(
            {'result': r'(\d+[+=-])[BW]'},
            {'result': r'\1'},
            regex=True
        )
        .assign(
            rank_2 = lambda x: x.result.str.slice(0, -1),
            W      = lambda x: x.result.str.slice(   -1)
        )
        .replace({'W': {    # W = number of wins, draws counting 1/2 (Elo, 1978)
            '+': 1.0,
            '=': 0.5,
            '-': 0.0,
            '?': np.nan
        }})
        .astype(dtype={column: int for column in ['rank_1', 'rank_2']})
        .pipe(lambda x: x
            .merge(x
                .loc[:, ['event_id', 'gid']]
                .drop_duplicates(subset=['event_id', 'gid'])
                .sort_values(['event_id', 'gid'])
                .reset_index(drop=True)
                .reset_index()
                .rename(columns={'index': 'group_id'})
            )
        )
        .loc[:, [
            'group_id', 'rank_1', 'rank_2',
            'round', 'unplayed', 'W'
        ]]
    )

def _names(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .sort_values('pid')
        .reset_index(drop=True)
        .rename(columns={
            'pid' : 'id'
        })
    )

def _games(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .pipe(lambda x: x
            .set_axis(x
                .columns
                .to_flat_index()
                .map('_'.join)
                , axis='columns', inplace=False
            )
        )
        .rename(columns=lambda x: x.strip('_'))
        .rename(columns=lambda x: x.replace(u'\xa0\u2191', ''))
        .rename(columns=lambda x: x.replace(' ', '_'))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: re.sub(r'event_(.*)', r'\1', x))
        .rename(columns=lambda x: re.sub(r'opponent_(.*)', r'\g<1>_2', x))
        .rename(columns=lambda x: re.sub(r'result_(.*)', r'\1', x))
        .rename(columns=lambda x: re.sub(r'(.*)name(.*)', r'\1\2', x))
        .rename(columns={
            'pid'      : 'player_id_1',
            'rating_2' : 'R_2',         # R = player 2's current rating
            'expected' : 'We',          # We = the expected score W (Elo, 1978)
            'observed' : 'W',           # W = the number of wins, draws counting 1/2 (Elo, 1978)
            'net_yield': 'dW'           # dW = W - We ('d' from 'delta')
        })
        .astype(dtype={
            'date': 'datetime64[ns]',
            'R_2'  : pd.Int64Dtype()
        })
        .loc[:, [
            'date', 'place', 'player_id_1', 'sur_2', 'pre_2',
            'R_2', 'significance',
            'unplayed', 'W', 'We', 'dW'
        ]]
    )

def _lists(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .pipe(lambda x: x
            .set_axis(pd
                .Index([
                    column[-1]
                    for column in x.columns.tolist()
                ])
                , axis='columns', inplace=False
            )
        )
        .drop(columns='variable_0')
        .reset_index()
        .rename(columns={
            'index'     : 'id',
            'variable_1': 'place',
            'variable_2': 'date',
            'variable_3': 'significance'
        })
        .astype(dtype={
            'date'        : 'datetime64[ns]',
            'significance': float
        })
        .loc[:, [
            'id', 'date', 'place', 'significance'
        ]]
        .sort_values('id', ascending=False)
        .reset_index(drop=True)
        .drop(columns=['id'])
    )

def _ratings(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .pipe(lambda x: x
            .set_axis(pd
                .Index([
                    column[2:]
                    for column in x.columns
                ])
                .to_flat_index()
                .map(dict.fromkeys)
                .map(list)
                .map('_'.join)
                , axis='columns', inplace=False
            )
        )
        .rename(columns=lambda x: x.strip('._'))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: re.sub(r'ranking(_)(\w{3}).*', r'\2\1rank', x))
        .rename(columns=lambda x: re.sub(r'(games)(_)(\w{3}).*', r'\3\2\1', x))
        .rename(columns=lambda x: re.sub(r'(.*)name', r'\1', x))
        .rename(columns={'rating': 'R'})
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
            'pre', 'sur', 'nat',
            'R', 'int_rank', 'nat_rank', 'eff_games', 'tot_games'
        ]]
    )

def _history(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .pipe(lambda x: x
            .set_axis(pd
                .Index([
                    column[-1]
                    for column in x.columns
                ])
                , axis='columns', inplace=False
            )
        )
        .drop(columns=['variable_0', 'variable_3'])
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: re.sub(r'(.*)name', r'\1', x))
        .rename(columns={
            'variable_1': 'place',
            'variable_2': 'date',
            'rating'    : 'R'
        })
        .replace({'R': {'unrated': np.nan}})
        .query('R.notnull()')
        .reset_index(drop=True)
        .astype(dtype={'date': 'datetime64[ns]'})
        .astype(dtype={'R'   : float           })
        .astype(dtype={'R'   : pd.Int64Dtype() })
        .loc[:, [
            'pre', 'sur', 'date', 'place',
            'R'
        ]]
        .pipe(lambda x: x
            .merge(x
                .loc[:, ['date', 'place']]
                .drop_duplicates()
                .reset_index()
                .rename(columns={'index': 'eid'})
                , how='left'
            )
        )
        .pipe(lambda x: x
            .merge(x
                .loc[:, ['pre', 'sur']]
                .drop_duplicates()
                .reset_index()
                .rename(columns={'index': 'rank'})
                , how='left'
            )
        )
        .sort_values(
            by=['eid', 'R', 'rank'],
            ascending=[False, False, True]
        )
        .reset_index(drop=True)
        .drop(columns=['eid', 'rank'])
    )
