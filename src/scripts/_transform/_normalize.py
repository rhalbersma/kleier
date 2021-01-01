#          Copyright Rein Halbersma 2019-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re
from typing import List

import numpy as np
import pandas as pd

def _is_key(df: pd.DataFrame, key: List[str]) -> bool:
    return not df.duplicated(subset=key).any()

def _has_consistent_index(df: pd.DataFrame) -> bool:
    return (
        df.index.is_monotonic_increasing and
        df.index.is_unique and
        df.index.min() == 0 and
        df.index.max() == df.shape[0] - 1
    )

def _tournaments(tournaments: pd.DataFrame) -> pd.DataFrame:
    df = tournaments
    key = ['eid']
    # 3NF
    assert _is_key(df, key)
    assert df.equals(df.sort_values(key))
    assert _has_consistent_index(df)
    return df

def _events(events: pd.DataFrame) -> pd.DataFrame:
    df = events
    key = ['eid']
    # 3NF
    assert _is_key(df, ['date', 'place'])
    assert _is_key(df, key)
    assert df.equals(df.sort_values(['date'] + key))
    assert df.equals(df.sort_values(key))
    assert _has_consistent_index(df)
    return df

def _groups(groups: pd.DataFrame) -> pd.DataFrame:
    df = groups
    key = ['eid', 'gid']
    # 3NF
    assert _is_key(df, key)
    assert df.equals(df.sort_values(key))
    assert _has_consistent_index(df)
    return df

def _names(names: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    key = ['pid' ]
    assert _is_key(names, ['name'])
    assert _is_key(names, key)
    df = (names
        .merge(standings
            .loc[:, ['pre', 'sur', 'nat']]
            .drop_duplicates()
            .assign(name = lambda x: x.pre + ' ' + x.sur)
            .assign(name = lambda x: x.name.str.strip())
            , how='left', on=['name'], validate='one_to_one'
        )
        .drop(columns='name')
    )
    # 3NF
    assert _is_key(df, ['pre', 'sur'])
    assert _is_key(df, key)
    assert df.equals(df.sort_values(key))
    assert _has_consistent_index(df)
    return df

def _dates(dates: pd.DataFrame) -> pd.DataFrame:
    old_key = ['date', 'place']
    assert _is_key(dates, old_key)
    key = ['date']
    attributes = [
        column
        for column in dates.columns.to_list()
        if not column in old_key
    ]
    df = (dates
        .loc[:, key + attributes]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # 3NF
    assert _is_key(df, key)
    assert df.equals(df.sort_values(key))
    assert _has_consistent_index(df)
    return df

def _ratings(ratings: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    old_key = ['pre', 'sur', 'nat']
    assert _is_key(ratings, old_key)
    key = ['pid']
    attributes = [
        column
        for column in ratings.columns.to_list()
        if not column in old_key
    ]
    df = (ratings
        .merge(names
            .loc[:, key + old_key]
            , how='left', on=old_key, validate='one_to_one'
        )
        .loc[:, key + ['nat'] + attributes]
    )
    # 2NF
    assert _is_key(df, key)
    assert df.equals(df.reset_index().sort_values(['R', 'index'], ascending=[False, True]).drop(columns=['index']))
    assert _has_consistent_index(df)
    # 3NF
    assert df.int_rank.isnull().equals(df.eff_games < 5.0)
    assert df.nat_rank.isnull().equals(df.eff_games < 5.0)
    df_eff = df.query('eff_games >= 5.0')
    df_nat = df_eff.groupby('nat')
    assert df_eff.int_rank.equals(df_eff.R.rank(ascending=False, method='first').astype('Int64'))
    assert df_eff.nat_rank.equals(df_nat.R.rank(ascending=False, method='first').astype('Int64'))
    df = df.drop(columns=['nat', 'int_rank', 'nat_rank'])
    return df

def _history(history: pd.DataFrame, events: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    old_key = ['date', 'place', 'pre', 'sur']
    assert _is_key(history, old_key)
    key = ['eid', 'pid']
    attributes = [
        column
        for column in history.columns.to_list()
        if not column in old_key
    ]
    df = (history
        .merge(events
            .loc[:, ['eid', 'date', 'place']]
            , how='left', on=['date', 'place'], validate='many_to_one'
        )
        .merge(names
            .loc[:, ['pid', 'pre', 'sur']]
            , how='left', on=['pre', 'sur'], validate='many_to_one'
        )
        .loc[:, key + attributes]
        .pipe(lambda x: x
            .merge(x
                .loc[:, ['pid']]
                .drop_duplicates(keep='first')
                .reset_index()
                , how='left'
            )
        )
        .sort_values(
            by=['eid', 'R', 'index'],
            ascending=[True, False, True]
        )
        .reset_index(drop=True)
        .drop(columns=['index'])
    )
    # 3NF
    assert _is_key(df, key)
    assert df.equals(df.reset_index().sort_values(['eid', 'R', 'index'], ascending=[True, False, True]).drop(columns=['index']))
    assert _has_consistent_index(df)
    return df

def _activity(activity: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    old_key = ['pre', 'sur', 'nat', 'eid']
    assert _is_key(activity, old_key)
    key = ['pid', 'eid']
    attributes = [
        column
        for column in activity.columns.to_list()
        if not column in old_key
    ]
    df = (activity
        .merge(names
            .loc[:, ['pid', 'pre', 'sur', 'nat']]
            , how='left', on=['pre', 'sur', 'nat'], validate='many_to_one'
        )
        .loc[:, key + attributes]
        .sort_values(key)
        .reset_index(drop=True)
    )
    # 3NF
    assert _is_key(df, key)
    assert df.equals(df.sort_values(key))
    assert _has_consistent_index(df)
    return df

def _standings(standings: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    old_key = ['eid', 'gid', 'pre', 'sur', 'nat']
    assert _is_key(standings, old_key)
    key = ['eid', 'gid', 'pid']
    attributes = [
        column
        for column in standings.columns.to_list()
        if not column in old_key
    ]
    df = (standings
        .merge(names
            .loc[:, ['pid', 'pre', 'sur', 'nat']]
            , how='left', on=['pre', 'sur', 'nat'], validate='many_to_one'
        )
        .loc[:, key + attributes]
    )
    # 3NF
    assert _is_key(df, ['eid', 'gid', 'rank'])
    assert df.equals(df.sort_values(['eid', 'gid', 'rank']))
    assert df.equals(df.sort_values(
        ['eid', 'gid', 'score', 'median', 'buchholz', 'dmr_W'],
        ascending=[True, True, False, False, False, False]
    ))
    assert _is_key(df, key)
    assert _has_consistent_index(df)
    return df

def _results(results: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    old_key = ['eid', 'gid', 'round', 'rank1', 'rank2']
    assert _is_key(results, old_key)
    key = ['eid', 'gid', 'round', 'pid1', 'pid2']
    attributes = [
        column
        for column in results.columns.to_list()
        if not column in old_key
    ]
    p1 = (standings
        .loc[:, ['eid', 'gid', 'rank', 'pid']]
        .rename(columns={
            'pid' : 'pid1',
            'rank': 'rank1'
        })
    )
    p2 = (standings
        .loc[:, ['eid', 'gid', 'rank', 'pid']]
        .rename(columns={
            'pid' : 'pid2',
            'rank': 'rank2'
        })
    )
    games = (results
        .query('rank2 != 0')
        .reset_index()
        .merge(p1, how='left', on=['eid', 'gid', 'rank1'], validate='many_to_one')
        .merge(p2, how='left', on=['eid', 'gid', 'rank2'], validate='many_to_one')
        .set_index('index', verify_integrity=True)
        .rename_axis(None)
    )
    dummy = (results
        .query('rank2 == 0')
        .reset_index()
        .merge(p1, how='left', on=['eid', 'gid', 'rank1'], validate='many_to_one')
        .assign(pid2 = 0)
        .set_index('index', verify_integrity=True)
        .rename_axis(None)
    )
    df = (pd
        .concat([games, dummy])
        .loc[:, key + attributes]
        .sort_index()
    )
    # 3NF
    assert _is_key(df, key)
    assert _has_consistent_index(df)
    return df

def _expected(expected: pd.DataFrame, events: pd.DataFrame, names: pd.DataFrame, dates: pd.DataFrame, ratings: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    old_key = ['date', 'place', 'pid1', 'pre2', 'sur2']
    key = ['eid', 'pid1', 'pid2']
    attributes = [
        column
        for column in expected.columns.to_list()
        if not column in old_key
    ]
    named = (expected
        .merge(events
            .loc[:, ['eid', 'date', 'place']]
            , how='left', on=['date', 'place'], validate='many_to_one'
        )
        .merge(names
            .add_suffix('2')
            .loc[:, ['pid2', 'pre2', 'sur2']]
            , how='left', on=['pre2', 'sur2'], validate='many_to_one'
        )
        .merge(ratings
            .loc[:, ['pid', 'R']]
            .add_suffix('1')
            , how='left', on=['pid1'], validate='many_to_one'
        )
        .merge(ratings
            .loc[:, ['pid', 'R']]
            .add_suffix('2')
            , how='left', on=['pid2', 'R2'], validate='many_to_one'
        )
        .loc[:, key + ['pre2', 'sur2', 'date', 'R1'] + attributes]
    )
    anonymous = (named
        .query('pre2.isnull() | sur2.isnull()')
        .loc[:, key + ['date', 'R1'] + attributes]
        .rename(columns=lambda x: re.sub(r'(.+)1', r'\g<1>0', x))
        .rename(columns=lambda x: re.sub(r'(.+)2', r'\g<1>1', x))
        .rename(columns=lambda x: re.sub(r'(.+)0', r'\g<1>2', x))
        .assign(
            W  = lambda x: 1.0 - x.W,
            We = lambda x: 1.0 - x.We,
            dW = lambda x: -x.dW
        )
        .loc[:, key + ['date', 'R1'] + attributes]
    )
    df = (named
        .drop(columns=['pre2', 'sur2'])
        .append(anonymous)
        .sort_values(
            by=['pid1', 'eid', 'R2'],
            ascending=[True, False, False]
        )
        .reset_index(drop=True)
    )
    # 1NF
    assert _has_consistent_index(df)
    # 2NF
    assert np.isclose(df.loc[:, ['date', 'significance']].drop_duplicates().significance.sort_values(), dates.significance).all()
    assert df.loc[:, ['date', 'significance']].drop_duplicates().drop(columns='significance').equals(df.loc[:, ['date'        ]].drop_duplicates())
    assert df.loc[:, ['pid1', 'R1'          ]].drop_duplicates().drop(columns='R1'          ).equals(df.loc[:, ['pid1'        ]].drop_duplicates())
    assert df.loc[:, ['pid2', 'R2'          ]].drop_duplicates().drop(columns='R2'          ).equals(df.loc[:, ['pid2'        ]].drop_duplicates())
    assert df.loc[:, ['pid1', 'pid2', 'We'  ]].drop_duplicates().drop(columns='We'          ).equals(df.loc[:, ['pid1', 'pid2']].drop_duplicates())
    # 3NF
    df0 = (results
        .query('pid2 != 0')
        .loc[:,      ['eid', 'pid1', 'pid2', 'W', 'unplayed']]
        .sort_values(['eid', 'pid1', 'pid2', 'W', 'unplayed'])
        .reset_index(drop=True)
    )
    df1 = (df
        .loc[:,      ['eid', 'pid1', 'pid2', 'W', 'unplayed']]
        .sort_values(['eid', 'pid1', 'pid2', 'W', 'unplayed'])
        .reset_index(drop=True)
    )
    assert ((df0.unplayed == df1.unplayed) | df1.unplayed.isnull()).all() and df0.W.equals(df1.W)
    assert (df.R1.isnull() | df.R2.isnull()).equals(df.We.isnull())
    assert np.isclose(df.dW, df.W - df.We, equal_nan=True).all()
    key = ['pid1', 'pid2']
    attributes = ['We']
    df = (df
        .loc[:, key + attributes]
        .drop_duplicates()
        .query('We.notnull()')
        .sort_values(key)
        .reset_index(drop=True)
    )
    assert _is_key(df, key)
    assert df.equals(df.sort_values(key))
    return df
