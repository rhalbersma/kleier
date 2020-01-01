#!/usr/bin/env python

#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import numpy as np
import pandas as pd
import scipy.stats as stats

import kleier.utils

import fmt_events
import fmt_players
#import fmt_ratings
import get_events
import get_players
import get_ratings

def normalize_players(players: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    assert not players.loc[:, ['name']].duplicated().any()
    pid_sur_pre_nat = (players
        .merge(standings
            .loc[:, ['sur', 'pre', 'nat']]
            .drop_duplicates()
            .assign(name = lambda x: x.pre + ' ' + x.sur)
            .assign(name = lambda x: x.name.str.strip())
            , how='outer', on=['name'], indicator=True, validate='one_to_one'
        )
    )
    df = (pid_sur_pre_nat
        .query('_merge == "both"')
        .drop(columns=['name', '_merge'])
        .append(pid_sur_pre_nat
            .query('_merge != "both"')
            .assign(
                sur = lambda x: x.name.str.split(expand=True)[1],
                pre = lambda x: x.name.str.split(expand=True)[0]
            )
            .drop(columns=['name', '_merge'])
        )
        .sort_values('id')
    )
    assert not df.loc[:, ['pre', 'sur']].duplicated().any()
    assert not df.loc[:, ['id'        ]].duplicated().any()
    assert df.equals(df.sort_values('id'))
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    return df

def normalize_standings(standings: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    assert not standings.loc[:, ['group_id', 'sur', 'pre', 'nat']].duplicated().any()
    df = (standings
        .merge(players
            .rename(columns={'id': 'player_id'})
            , how='left', on=['sur', 'pre', 'nat'], validate='many_to_one'
        )
        .loc[:, [
            'group_id', 'player_id', 
            'rank', 'score', 'buchholz', 'median', 'dmr_W', 'dmr_N',
            'eff_games', 'Ro', 'dR', 'Rn'
        ]]
    )
    assert not df.loc[:, ['group_id', 'player_id']].duplicated().any()
    assert not df.loc[:, ['group_id', 'rank'     ]].duplicated().any()
    assert df.equals(df.sort_values(
        ['group_id', 'score', 'median', 'buchholz', 'dmr_W'], 
        ascending=[True, False, False, False, False]
    ))    
    assert df.equals(df.sort_values(['group_id', 'rank']))
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    return df

def normalize_results(results: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
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
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    return df

def normalize_games(games: pd.DataFrame, events: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    df = (games
        .merge(events
            .loc[:, ['id', 'date', 'place']]
            .rename(columns={'id': 'event_id'})
            , how='left', on=['date', 'place'], validate='many_to_one'
        )
        .merge(players
            .loc[:, ['id', 'sur', 'pre']]
            .add_suffix('_2')
            .rename(columns={'id_2': 'player_id_2'})
            , how='left', on=['sur_2', 'pre_2'], validate='many_to_one'
        )
        .loc[:, [
            'event_id', 'player_id_1', 'player_id_2',
            'R_2', 'significance', 'unplayed', 'W', 'We', 'dW'
        ]]
    )

def _group_format(M: pd.Series, N: pd.Series) -> pd.Series:
    return np.where(
        N == M - 1 + M % 2,
        'RR1',
        np.where(
            N == 2 * (M - 1 + M % 2),
            'RR2',
            np.where(
                N >= M,
                'RRX',
                'SS'
            )
        )
    )

def add_groups_format(groups: pd.DataFrame) -> pd.DataFrame:
    return (groups
        .assign(format = lambda x: _group_format(x.M, x.N))
        .loc[:, [
            'id', 'event_id', 
            'name', 'group', 'score_W', 'score_D', 'score_L', 
            'M', 'N', 'format',
            'file_from', 'file_date', 'file_name', 'remarks'
        ]]
    )


def merge_games_pinf(games: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    return (games
        .merge(players
            .rename(columns={'id': 'player_id'})
            .add_suffix('_1'),
            how='left', validate='many_to_one'
        )
        .merge(players
            .rename(columns={'id': 'player_id'})
            .add_suffix('_2'),
            how='left', validate='many_to_one'
        )
        .loc[:, [
            'id', 'date', 'place',
            'player_id_1', 'sur_1', 'pre_1', 'nat_1',
            'player_id_2', 'sur_2', 'pre_2', 'nat_2', 
            'R_2', 'significance', 
            'unplayed', 'W', 'We', 'dW'
        ]]
    )

def append_games_anonymous(games: pd.DataFrame) -> pd.DataFrame:
    return (games
        .append(games
            .query('sur_2.isnul() & pre_2.isnul()')
            .loc[:, [
                'date', 'place', 'unplayed',
                'pid2', 'sur2', 'pre2', 'pnat2', 'R2',
                'pid1', 'sur1', 'pre1', 'pnat1', 'R1',
                'significance', 'W', 'We', 'dW'
            ]]
            .rename(columns=lambda x: re.sub(r'(.+)1', r'\g<1>0', x))
            .rename(columns=lambda x: re.sub(r'(.+)2', r'\g<1>1', x))
            .rename(columns=lambda x: re.sub(r'(.+)0', r'\g<1>2', x))
            .assign(
                W  = lambda x: 1.0 - x.W,
                We = lambda x: 1.0 - x.We,
                dW = lambda x: -x.dW
            )
        )
        .sort_values(
            by=['player_id_1', 'date', 'R_2'],
            ascending=[True, False, False]
        )
        .reset_index(drop=True)
    )

def merge_standings_einf_pinf(standings: pd.DataFrame, events: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    return (events
        .loc[:, ['eid', 'gid', 'date', 'place', 'enat']]
        .merge(standings, how='left', validate='one_to_many')
        .merge(players
            .drop(columns=['R']),
            how='left', validate='many_to_one'
        )
        .loc[:, [
            'eid', 'gid', 'date', 'place', 'enat',
            'rank', 'pid', 'sur', 'pre', 'pnat',
            'Ro', 'dR', 'Rn', 'eff_games',
            'score', 'buchholz', 'median', 'compa'
        ]]
    )

def merge_results_einf_pinf(results: pd.DataFrame, events: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    return (events
        .loc[:, ['eid', 'gid', 'date', 'place', 'enat']]
        .merge(results, how='left', validate='one_to_many')
        .merge(standings
            .loc[:, [
                'eid', 'gid', 'date', 'place', 'enat',
                'rank', 'pid', 'sur', 'pre', 'pnat',
                'Ro', 'Rn'
            ]]
            .rename(columns={x: x + '1' for x in ['rank', 'pid', 'sur', 'pre', 'pnat', 'Ro', 'Rn']}),
            how='left', validate='many_to_one'
        )
        .merge(standings
            .loc[:, [
                'eid', 'gid', 'date', 'place', 'enat',
                'rank', 'pid', 'sur', 'pre', 'pnat',
                'Ro', 'Rn'
            ]]
            .rename(columns={x: x + '2' for x in ['rank', 'pid', 'sur', 'pre', 'pnat', 'Ro', 'Rn']}),
            how='outer', validate='many_to_one', indicator=True
        )
        .query('_merge != "right_only"')
        .drop(columns=['_merge'])
        .sort_values(by=['eid', 'gid', 'round', 'rank1'])
        .reset_index(drop=True)
        .fillna({'pid2': 0})
        .astype(dtype={column: int for column in ['round', 'rank1', 'pid1', 'pid2']})
        .loc[:, [
            'eid', 'gid', 'date', 'place', 'enat', 'round', 'unplayed',
            'rank1', 'pid1', 'sur1', 'pre1', 'pnat1', 'Ro1', 'Rn1',
            'rank2', 'pid2', 'sur2', 'pre2', 'pnat2', 'Ro2', 'Rn2',
            'result'
        ]]
    )

def merge_results_games(results: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    return (results
        .merge(games
            .loc[:, ['date', 'place', 'significance']]
            .drop_duplicates(['date', 'place'])
        )
        .rename(columns={'result': 'W'})
        .replace({'W': {'+': 1.0, '=': 0.5, '-': 0.0}})
        .merge(games
            .loc[:, ['date', 'place', 'unplayed', 'pid1', 'R1', 'pid2', 'R2', 'W', 'We', 'dW']]
            .drop_duplicates(),
            how='outer', validate='many_to_many', indicator=True
        )
        .loc[:, [
            'eid', 'gid', 'date', 'place', 'enat', 'significance', 'unplayed', 'round',
            'rank1', 'pid1', 'sur1', 'pre1', 'pnat1', 'R1', 'Ro1', 'Rn1',
            'rank2', 'pid2', 'sur2', 'pre2', 'pnat2', 'R2', 'Ro2', 'Rn2',
            'W', 'We', 'dW', '_merge'
        ]]
    )

def event_cross_add_outcome(event_cross: pd.DataFrame) -> pd.DataFrame:
    return (event_cross
        .assign(
            outcome = lambda x: np.where(
                x.rank2 != 0,
                x.result.map({
                    '+': 'W',
                    '=': 'D',
                    '-': 'L'}),
                x.result.map({
                    '+': 'B',
                    '-': 'F'})))
        .astype(dtype={'outcome': 'category'})
        .assign(outcome = lambda x: x.outcome.cat.set_categories(
            ['W', 'D', 'L', 'B', 'F'])))

def event_table_add_outcome(event_table: pd.DataFrame, event_cross: pd.DataFrame) -> pd.DataFrame:
    grouper = ['eid', 'gid', 'rank1']
    games = (event_cross
        .groupby(grouper)
        .agg(G = ('outcome', 'count'))
        .reset_index())
    played = (event_cross
        .query('rank2 != 0')
        .groupby(grouper)
        .agg(P = ('outcome', 'count'))
        .reset_index())
    outcome = (event_cross
        .groupby(grouper)
        ['outcome']
        .value_counts()
        .unstack(fill_value=0)
        .reset_index())
    outcome = outcome[grouper + event_cross['outcome'].cat.categories.to_list()]
    counts = (pd
        .merge(games, played, how='outer')
        .fillna({'P': 0})
        .astype(dtype={'P': int})
        .merge(outcome)
        .rename(columns={'rank1': 'rank'}))
    return (event_table
        .merge(counts))

def event_cross_add_points(event_cross: pd.DataFrame, event_index: pd.DataFrame) -> pd.DataFrame:
    return (event_index
        .drop(columns=['name', 'group'])
        .merge(event_cross)
        .assign(
            score = lambda x: np.select(
                [x.result == '+', x.result == '=', x.result == '-'],
                [x.W,             x.D,             x.L            ]),
            buch = lambda x: np.select(
                [x.outcome == 'W', x.outcome == 'D', x.outcome == 'L'],
                [x.W,              x.D,              x.L,            ],
                (x.W + x.L) / 2),
            pts = lambda x: x.outcome.map({
                'W': 1.0,
                'D': 0.5,
                'L': 0.0,
                'B': 0.0,
                'F': 0.0}))
            .drop(columns=['W', 'D', 'L']))

def event_table_add_points(event_table: pd.DataFrame, event_cross: pd.DataFrame) -> pd.DataFrame:
    return (event_table
        .merge(event_cross
            .rename(columns={'rank1': 'rank'})
            .groupby(['eid', 'gid', 'rank'])
            .agg(
                buch = ('buch', 'sum'),
                pts = ('pts', 'sum'))
            .reset_index())
        .assign(p  = lambda x: x.pts / x.P))

def event_cross_ratings(event_cross: pd.DataFrame) -> pd.DataFrame:
    df = event_cross
    df = (df
        .query('rank2 != 0')
        .assign(
            R = lambda x: x.rating2.where(x.rating2 >= rating_floor, rating_floor),
            p = lambda x: x.outcome.map({
                'W': 1.0,
                'D': 0.5,
                'L': 0.0})
        )
        .sort_values(
            by=       ['eid', 'gid', 'rank1', 'R',    'rank2'],
            ascending=[ True,  True,  True,    False,  True]))
    df2 = (df
        .groupby(['eid', 'gid', 'rank1', 'pid1', 'sur1', 'pre1', 'nat1'])
        .agg(
            n  = ('R', 'count'),
            p  = ('p', 'mean'),
            Ra = ('R', 'mean'))
        .reset_index()
        .round({'Ra': 0})
        .assign(
            dp = lambda x: dp(x.n, x.p),
            Rp = lambda x: x.Ra + x.dp)
        .astype(dtype={
            'Ra': int,
            'dp': int,
            'Rp': int}))

def main():
    # Complete Kleier archive as of 2019-12-15
    # now = pd.to_datetime('2019-12-15')
    # max_eid, max_pid = 670, 2970
    # get_events.main(max_eid)
    # get_players.main(max_pid)
    # get_ratings.main(max_eid, max_pid)

    events      = kleier.load_dataset('events'   ).pipe(fmt_events.format_events   )
    groups      = kleier.load_dataset('groups'   ).pipe(fmt_events.format_groups   )
    standings   = kleier.load_dataset('standings').pipe(fmt_events.format_standings)
    results     = kleier.load_dataset('results'  ).pipe(fmt_events.format_results  )
    players     = kleier.load_dataset('players'  ).pipe(fmt_players.format_players )
    games       = kleier.load_dataset('games'    ).pipe(fmt_players.format_games   )    
    ratings     = kleier.load_dataset('ratings'  ) # TODO: formatting

    assert not events.loc[:, ['place', 'date']].duplicated().any()
    assert not groups.drop(columns='id').duplicated().any()
    
    players     = normalize_players(players, standings)
    standings   = normalize_standings(standings, players)
    results     = normalize_results(results, standings)
    
    games       = normalize_games(games, players)

    groups      = add_groups_format(groups)    
    games       = merge_games_pinf(games, players)

    games       = append_games_anonymous(games)
    standings   = merge_standings_einf_pinf(standings, events, players)
    results     = merge_results_einf_pinf(results, events, standings)
    results     = merge_results_games(results, games)


    event_cross = event_cross_add_outcome(event_cross)
    event_table = event_table_add_outcome(event_table, event_cross)

    event_cross = event_cross_add_points(event_cross, event_index)
    event_table = event_table_add_points(event_table, event_cross)

