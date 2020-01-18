#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import numpy as np
import pandas as pd
from typing import Tuple

def format_events(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .loc[:, [
            'eid', 'date', 'place', 'nat'
        ]]
        .sort_values('eid')
        .reset_index(drop=True)
        .rename(columns={'eid' : 'id'})
    )

def format_groups(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .sort_values(['eid', 'gid'])
        .reset_index()
        .drop(columns=['gid'])
        .rename(columns={
            'index': 'id',
            'eid'  : 'event_id'
        })
    )

def format_standings(df: pd.DataFrame) -> Tuple[pd.DataFrame]:
    standings_activity = (df
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
            'nationality': 'nat',
            'value'      : 'Rn',            # Rn = a player's rating after a performance (Elo, 1978)
            'change'     : 'dR'             # dR = Rn - Ro ('d' from 'delta')
        })
        .assign(
            dmr_W = lambda x: np.where(x.compa.isnull(), 0, x.compa.str.split('/').str[0]),
            dmr_N = lambda x: np.where(x.compa.isnull(), 0, x.compa.str.split('/').str[1])
        )
        .astype(dtype={column: int             for column in ['rank', 'score', 'dmr_W', 'dmr_N']})
        .astype(dtype={column: float           for column in ['Rn', 'dR']})
        .astype(dtype={column: pd.Int64Dtype() for column in ['Rn', 'dR']})
        .astype(dtype={column: float           for column in ['eff_games', 'median', 'buchholz']})
        .assign(Ro = lambda x: x.Rn - x.dR) # Ro = a player's rating before a performance (Elo, 1978)
    )
    standings = (standings_activity
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
    activity = (standings_activity
        .loc[:, [
            'pre', 'sur', 'nat', 'event_id',
            'eff_games', 'Rn', 'Ro', 'dR'
        ]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return standings, activity

def format_results(df: pd.DataFrame) -> pd.DataFrame:
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
