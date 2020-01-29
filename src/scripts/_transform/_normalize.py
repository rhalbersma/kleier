#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import numpy as np
import pandas as pd

def _has_consistent_index(df: pd.DataFrame) -> bool:
    return (
        df.index.is_monotonic_increasing and
        df.index.is_unique and
        df.index.min() == 0 and
        df.index.max() == df.shape[0] - 1
    )

def _tournaments_byplace(tournaments_byplace: pd.DataFrame) -> pd.DataFrame:
    df = tournaments_byplace
    assert not df.loc[:, ['eid']].duplicated().any()
    assert df.equals(df.sort_values('eid'))
    assert _has_consistent_index(df)
    return df

def _events(events: pd.DataFrame) -> pd.DataFrame:
    df = events
    assert not df.loc[:, ['place', 'date']].duplicated().any()
    assert not df.loc[:, ['id'           ]].duplicated().any()
    assert df.equals(df.sort_values('id'))
    assert _has_consistent_index(df)
    return df

def _groups(groups: pd.DataFrame) -> pd.DataFrame:
    df = groups
    assert not df.loc[:, ['id'    ]].duplicated().any()
    assert df.equals(df.sort_values('id'))
    assert _has_consistent_index(df)
    return df

def _names(names: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    assert not names.loc[:, ['name']].duplicated().any()
    pid_sur_pre_nat = (names
        .merge(standings
            .loc[:, ['pre', 'sur', 'nat']]
            .drop_duplicates()
            .assign(name = lambda x: x.pre + ' ' + x.sur)
            .assign(name = lambda x: x.name.str.strip())
            , how='outer', on='name', indicator=True, validate='one_to_one'
        )
    )
    df = (pid_sur_pre_nat
        .query('_merge == "both"')
        .drop(columns=['name', '_merge'])
        .append(pid_sur_pre_nat
            .query('_merge != "both"')
            .assign(
                pre = lambda x: x.name.str.split(expand=True)[0],
                sur = lambda x: x.name.str.split(expand=True)[1]
            )
            .drop(columns=['name', '_merge'])
        )
        .sort_values('id')
    )
    assert not df.loc[:, ['pre', 'sur']].duplicated().any()
    assert not df.loc[:, ['id'        ]].duplicated().any()
    assert df.equals(df.sort_values('id'))
    assert _has_consistent_index(df)
    return df

def _ratings(ratings: pd.DataFrame, events: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    assert not ratings.loc[:, ['pre', 'sur', 'nat']].duplicated().any()
    df = (ratings
        .merge(names
            .rename(columns={'id': 'player_id'})
            , how='left', on=['pre', 'sur', 'nat'], validate='many_to_one'
        )
        .loc[:, [
            'player_id',
            'R', 'int_rank', 'nat_rank',
            'eff_games', 'tot_games'
        ]]
    )
    assert not df.loc[:, ['player_id']].duplicated().any()
    assert df.equals(df.reset_index().sort_values(['R', 'index'], ascending=[False, True]).drop(columns=['index']))
    assert _has_consistent_index(df)
    return df

def _history(history: pd.DataFrame, events: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    assert not history.loc[:, ['date', 'place', 'pre', 'sur']].duplicated().any()
    df = (history
        .merge(events
            .loc[:, ['id', 'date', 'place']]
            .rename(columns={'id': 'event_id'})
            , how='left', on=['date', 'place'], validate='many_to_one'
        )
        .merge(names
            .rename(columns={'id': 'player_id'})
            , how='left', on=['pre', 'sur'], validate='many_to_one'
        )
        .rename(columns={'R': 'Rn'})
        .loc[:, [
            'event_id', 'player_id', 'Rn'
        ]]
        .sort_values(['player_id', 'event_id'])
        .groupby('player_id')
        .apply(lambda p: p.assign(Ro = lambda x: x.Rn.shift(1)))
        .sort_index()
        .assign(dR = lambda x: x.Rn - x.Ro)
    )
    assert not df.loc[:, ['event_id', 'player_id']].duplicated().any()
    assert df.equals(df.reset_index().sort_values(['event_id', 'Rn', 'index'], ascending=[True, False, True]).drop(columns=['index']))
    assert _has_consistent_index(df)
    return df

def _lists(lists: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    assert not lists.loc[:, ['date', 'place']].duplicated().any()
    df = (lists
        .merge(events
            .loc[:, ['id', 'date', 'place']]
            .rename(columns={'id': 'event_id'})
            , how='left', on=['date', 'place'], validate='one_to_one'
        )
        .loc[:, [
            'event_id', 'significance'
        ]]
    )
    assert not df.loc[:, ['event_id']].duplicated().any()
    assert df.equals(df.sort_values('event_id'))
    assert _has_consistent_index(df)
    return df

def _standings(standings: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    assert not standings.loc[:, ['group_id', 'pre', 'sur', 'nat']].duplicated().any()
    df = (standings
        .merge(names
            .rename(columns={'id': 'player_id'})
            , how='left', on=['pre', 'sur', 'nat'], validate='many_to_one'
        )
        .loc[:, [
            'group_id', 'player_id',
            'rank', 'score', 'median', 'buchholz', 'dmr_W', 'dmr_N'
        ]]
    )
    assert not df.loc[:, ['group_id', 'player_id']].duplicated().any()
    assert not df.loc[:, ['group_id', 'rank'     ]].duplicated().any()
    assert df.equals(df.sort_values(
        ['group_id', 'score', 'median', 'buchholz', 'dmr_W'],
        ascending=[True, False, False, False, False]
    ))
    assert df.equals(df.sort_values(['group_id', 'rank']))
    assert _has_consistent_index(df)
    return df

def _activity(activity: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    assert not activity.loc[:, ['pre', 'sur', 'nat', 'event_id']].duplicated().any()
    df = (activity
        .merge(names
            .rename(columns={'id': 'player_id'})
            , how='left', on=['pre', 'sur', 'nat'], validate='many_to_one'
        )
        .loc[:, [
            'player_id', 'event_id',
            'eff_games', 'Rn', 'Ro', 'dR'
        ]]
        .sort_values(['player_id', 'event_id'])
        .reset_index(drop=True)
    )
    assert not df.loc[:, ['player_id', 'event_id']].duplicated().any()
    assert df.equals(df.sort_values(['player_id', 'event_id']))
    assert _has_consistent_index(df)
    return df

def _results(results: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    p1 = (standings
        .loc[:, ['group_id', 'player_id', 'rank']]
        .add_suffix('_1')
        .rename(columns={'group_id_1': 'group_id'})
    )
    p2 = (standings
        .loc[:, ['group_id', 'player_id', 'rank']]
        .add_suffix('_2')
        .rename(columns={'group_id_2': 'group_id'})
    )
    games = (results
        .query('rank_2 != 0')
        .reset_index()
        .merge(p1, how='left', on=['group_id', 'rank_1'], validate='many_to_one')
        .set_index('index', verify_integrity=True)
        .reset_index()
        .merge(p2, how='left', on=['group_id', 'rank_2'], validate='many_to_one')
        .set_index('index', verify_integrity=True)
        .rename_axis(None)
    )
    dummy = (results
        .query('rank_2 == 0')
        .reset_index()
        .merge(p1, how='left', on=['group_id', 'rank_1'], validate='many_to_one')
        .assign(player_id_2 = 0)
        .set_index('index', verify_integrity=True)
        .rename_axis(None)
    )
    df = (pd
        .concat([games, dummy])
        .loc[:, [
            'group_id', 'player_id_1', 'player_id_2',
            'round', 'unplayed', 'W'
        ]]
        .sort_index()
    )
    assert not df.loc[:, ['group_id', 'player_id_1', 'round']].duplicated().any()
    assert _has_consistent_index(df)
    return df

def _games(games: pd.DataFrame, events: pd.DataFrame, names: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    named = (games
        .merge(events
            .loc[:, ['id', 'date', 'place']]
            .rename(columns={'id': 'event_id'})
            , how='left', on=['date', 'place'], validate='many_to_one'
        )
        .merge(names
            .loc[:, ['id', 'pre', 'sur']]
            .add_suffix('_2')
            .rename(columns={'id_2': 'player_id_2'})
            , how='left', on=['sur_2', 'pre_2'], validate='many_to_one'
        )
        .merge(ratings
            .loc[:, ['player_id', 'R']]
            .add_suffix('_1')
            , how='left', on='player_id_1', validate='many_to_one'
        )
        .merge(ratings
            .loc[:, ['player_id', 'R']]
            .add_suffix('_2')
            , how='left', on=['player_id_2', 'R_2'], validate='many_to_one'
        )
        .loc[:, [
            'event_id', 'player_id_1', 'player_id_2', 'sur_2', 'pre_2',
            'significance', 'R_1', 'R_2', 'unplayed', 'W', 'We', 'dW'
        ]]
    )
    anonymous = (named
        .query('sur_2.isnull() & pre_2.isnull()')
        .loc[:, [
            'event_id', 'player_id_2', 'player_id_1',
            'significance', 'R_2', 'R_1', 'unplayed', 'W', 'We', 'dW'
        ]]
        .rename(columns=lambda x: re.sub(r'(.+_)1', r'\g<1>0', x))
        .rename(columns=lambda x: re.sub(r'(.+_)2', r'\g<1>1', x))
        .rename(columns=lambda x: re.sub(r'(.+_)0', r'\g<1>2', x))
        .assign(
            W  = lambda x: 1.0 - x.W,
            We = lambda x: 1.0 - x.We,
            dW = lambda x: -x.dW
        )
    )
    df = (named
        .drop(columns=['sur_2', 'pre_2'])
        .append(anonymous)
        .sort_values(
            by=['player_id_1', 'event_id', 'R_2'],
            ascending=[True, False, False]
        )
        .reset_index(drop=True)
    )
    assert df.equals(df.sort_values(['player_id_1', 'event_id', 'R_2'], ascending=[True, False, False]))
    assert _has_consistent_index(df)
    return df
