#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import numpy as np
import pandas as pd

def _tournaments(tournaments: pd.DataFrame) -> pd.DataFrame:
    key = ['eid']
    attributes = ['nat']
    return (tournaments
        .loc[:, key + attributes]
        .sort_values(key)
        .reset_index(drop=True)
    )

def _events(events: pd.DataFrame) -> pd.DataFrame:
    key = ['eid']
    attributes = ['date', 'place']
    assert events.equals(events.sort_values(key))
    return (events
        .loc[:, key + attributes]
    )

def _groups(groups: pd.DataFrame) -> pd.DataFrame:
    key = ['eid', 'gid']
    attributes = [
        'name', 'group', 'score_W', 'score_D', 'score_L', 'M', 'N',
        'file_from', 'file_date', 'file_name', 'remarks'
    ]
    assert groups.equals(groups.sort_values(key))
    return (groups
        .loc[:, key + attributes]
    )

def _activity(activity: pd.DataFrame) -> pd.DataFrame:
    key = ['pre', 'sur', 'nat', 'eid']
    attributes = ['R', 'dR', 'eff_games']
    return (activity
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
            'nationality': 'nat',
            'value'      : 'R',     # R  = a player's rating after a performance (Elo, 1978)
            'change'     : 'dR'     # dR = a player's chaing in rating after a performance ('d' from 'delta')
        })
        .astype(dtype={column: float   for column in ['R', 'dR']})
        .astype(dtype={column: 'Int64' for column in ['R', 'dR']})
        .astype(dtype={column: float   for column in ['eff_games']})
        .loc[:, key + attributes]
        .drop_duplicates()
        .reset_index(drop=True)
    )

def _standings(standings: pd.DataFrame) -> pd.DataFrame:
    key = ['eid', 'gid', 'pre', 'sur', 'nat']
    attributes = ['rank', 'score', 'median', 'buchholz', 'dmr_W', 'dmr_N']
    return (standings
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
            '#'          : 'rank',
            'nationality': 'nat'
        })
        .assign(
            dmr_W = lambda x: np.where(x.compa.isnull(), 0, x.compa.str.split('/').str[0]),
            dmr_N = lambda x: np.where(x.compa.isnull(), 0, x.compa.str.split('/').str[1])
        )
        .astype(dtype={column: int   for column in ['rank', 'score', 'dmr_W', 'dmr_N']})
        .astype(dtype={column: float for column in ['median', 'buchholz']})
        .loc[:, key + attributes]
    )

def _results(results: pd.DataFrame) -> pd.DataFrame:
    key = ['eid', 'gid', 'round', 'rank1', 'rank2']
    attributes = ['unplayed', 'W']
    return (results
        .rename(columns=lambda x: x.lower())
        .rename(columns={
            '##'     : 'rank1',
            'results': 'result'
        })
        .fillna({'result': '0?'})
        .replace(
            {'result': r'(\d+[+=-])[BW]'},
            {'result': r'\1'},
            regex=True
        )
        .assign(
            rank2 = lambda x: x.result.str.slice(0, -1),
            W     = lambda x: x.result.str.slice(   -1)
        )
        .replace({'W': {    # W = number of wins, draws counting 1/2 (Elo, 1978)
            '+': 1.0,
            '=': 0.5,
            '-': 0.0,
            '?': np.nan
        }})
        .astype(dtype={column: int for column in ['rank1', 'rank2']})
        .loc[:, key + attributes]
    )

def _names(names: pd.DataFrame) -> pd.DataFrame:
    key = ['pid']
    attributes = ['name']
    assert names.equals(names.sort_values(key))
    return (names
        .loc[:, key + attributes]
    )

def _expected(expected: pd.DataFrame) -> pd.DataFrame:
    key = ['date', 'place', 'pid1', 'pre2', 'sur2']
    attributes = ['R2', 'significance', 'unplayed', 'W', 'We', 'dW']
    return (expected
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
        .rename(columns=lambda x: re.sub(r'opponent_(.*)', r'\g<1>2', x))
        .rename(columns=lambda x: re.sub(r'result_(.*)', r'\1', x))
        .rename(columns=lambda x: re.sub(r'(.*)name(.*)', r'\1\2', x))
        .rename(columns={
            'pid'      : 'pid1',
            'rating2'  : 'R2',  # R = player 2's current rating
            'expected' : 'We',  # We = the expected score W (Elo, 1978)
            'observed' : 'W',   # W = the number of wins, draws counting 1/2 (Elo, 1978)
            'net_yield': 'dW'   # dW = W - We ('d' from 'delta')
        })
        .astype(dtype={
            'date': 'datetime64[ns]',
            'R2'  : 'Int64'
        })
        .loc[:, key + attributes]
    )

def _dates(dates: pd.DataFrame) -> pd.DataFrame:
    key = ['date', 'place']
    attributes = ['significance']
    return (dates
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
        .rename(columns={
            'variable_1': 'place',
            'variable_2': 'date',
            'variable_3': 'significance'
        })
        .astype(dtype={
            'date'        : 'datetime64[ns]',
            'significance': float
        })
        .loc[:, key + attributes]
        .sort_index(ascending=False)
        .reset_index(drop=True)
    )

def _ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    key = ['pre', 'sur', 'nat']
    attributes = ['R', 'int_rank', 'nat_rank', 'eff_games', 'tot_games']
    return (ratings
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
        .astype(dtype={column: float   for column in ['R', 'int_rank', 'nat_rank']})
        .astype(dtype={column: 'Int64' for column in ['R', 'int_rank', 'nat_rank']})
        .loc[:, key + attributes]
    )

def _history(history: pd.DataFrame) -> pd.DataFrame:
    key = ['date', 'place', 'pre', 'sur']
    attributes = ['R']
    return (history
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
        .astype(dtype={'R'   : 'Int64'         })
        .loc[:, key + attributes]
    )
