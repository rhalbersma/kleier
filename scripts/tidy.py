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

import get_events
import get_players
import get_ratings

def add_events_format(events: pd.DataFrame) -> pd.DataFrame:
    return (events
        .assign(format = lambda x:
            np.where(
                x.N == x.M - 1 + x.M % 2,
                'RR1',
                np.where(
                    x.N == 2 * (x.M - 1 + x.M % 2),
                    'RR2',
                    np.where(
                        x.N >= x.M,
                        'RRX',
                        'SS'
                    )
                )
            )
        )
        .loc[:, [
            'eid', 'gid', 'date', 'place', 'enat',
            'pW', 'pD', 'pL', 'M', 'N', 'format',
            'group', 'name', 'file_from', 'file_date', 'file_name', 'remarks'
        ]]
    )

def merge_players_pinf(players: pd.DataFrame, standings: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    pid_sur_pre_pnat = (players
        .merge(standings
            .loc[:, ['sur', 'pre', 'pnat']]
            .drop_duplicates()
            .assign(name = lambda x: x.pre + ' ' + x.sur)
            .assign(name = lambda x: x.name.str.strip()),
            how='outer', indicator=True, validate='one_to_one'
        )
    )
    return (pid_sur_pre_pnat
        .query('_merge == "both"')
        .drop(columns=['name', '_merge'])
        .append(pid_sur_pre_pnat
            .query('_merge != "both"')
            .assign(
                sur = lambda x: x.name.str.split(expand=True)[1],
                pre = lambda x: x.name.str.split(expand=True)[0]
            )
            .fillna({'pnat': ''})
            .drop(columns=['name', '_merge'])
        )
        .sort_values('pid')
        .merge(games
            .loc[:, games.columns.str.endswith('2')]
            .drop_duplicates()
            .rename(columns=lambda x: re.sub(r'(.*)2', r'\1', x)),
            how='outer', validate='one_to_one'
        )
    )

def merge_games_pinf(games: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    return (games
        .merge(players
            .rename(columns=lambda x: x + '1'),
            how='left', validate='many_to_one'
        )
        .merge(players
            .rename(columns=lambda x: x + '2'),
            how='left', validate='many_to_one'
        )
        .loc[:, [
            'date', 'place', 'unplayed',
            'pid1', 'sur1', 'pre1', 'pnat1', 'R1',
            'pid2', 'sur2', 'pre2', 'pnat2', 'R2',
            'significance', 'W', 'We', 'dW'
        ]]
    )

def append_games_anonymous(games: pd.DataFrame) -> pd.DataFrame:
    return (games
        .append(games
            .query('sur2 == "" & pre2 == ""')
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
            by=['pid1', 'date', 'R2'],
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
    # Complete Kleier archive as of 2019-11-16
    max_eid, max_pid = 667, 2965
    # get_events.main(max_eid)
    # get_players.main(max_pid)
    # get_ratings.main(max_eid, max_pid)

    events      = kleier.load_dataset('events').pipe(get_events.format_events)
    standings   = kleier.load_dataset('standings').pipe(get_events.format_standings)
    results     = kleier.load_dataset('results').pipe(get_events.format_results)
    players     = kleier.load_dataset('players')
    games       = kleier.load_dataset('games').pipe(get_players.format_games)
    ratings     = kleier.load_dataset('ratings').pipe(lambda x: get_ratings.format_ratings(x, max_eid))

    events      = add_events_format(events)
    players     = merge_players_pinf(players, standings, games)
    games       = merge_games_pinf(games, players)
    games       = append_games_anonymous(games)
    standings   = merge_standings_einf_pinf(standings, events, players)
    results     = merge_results_einf_pinf(results, events, standings)
    results     = merge_results_games(results, games)

    now = pd.to_datetime('2019-11-16')
    lam = 2.5731
    year = 365.2425
    df = (games
        .assign(dd = lambda x: (now - x.date).dt.days)
        .assign(sig = lambda x: np.round(np.exp(-(x.dd/year / lam)**2), 6))
    )[['date', 'significance', 'dd', 'sig']].drop_duplicates().sort_values('date')

    event_cross  = event_cross_add_outcome(event_cross)
    event_table  = event_table_add_outcome(event_table, event_cross)

    event_cross  = event_cross_add_points(event_cross, event_index)
    event_table  = event_table_add_points(event_table, event_cross)

