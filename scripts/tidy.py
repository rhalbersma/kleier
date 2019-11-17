#!/usr/bin/env python

#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import numpy as np
import pandas as pd

import kleier.utils

import get_events
import get_players

def merge_players_pinf(players: pd.DataFrame, standings: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    sur_pre_nat_name = (standings
        .loc[:, ['surname', 'prename', 'nationality']]
        .drop_duplicates()
        .assign(name = lambda x: x.prename + ' ' + x.surname)
        .assign(name = lambda x: x.name.str.strip())
    )
    pid_sur_pre_nat = (players
        .merge(
            sur_pre_nat_name,
            how='outer', indicator=True, validate='one_to_one'
        )
    )
    pid_sur_pre_no_nat = (pid_sur_pre_nat
        .query('_merge != "both"')
        .assign(
            surname = lambda x: x.name.str.split(expand=True)[1],
            prename = lambda x: x.name.str.split(expand=True)[0]
        )
        .fillna({'nationality': ''})
        .drop(columns=['name', '_merge'])
    )
    pid_sur_pre_nat = (pid_sur_pre_nat
        .query('_merge == "both"')
        .drop(columns=['name', '_merge'])
        .append(pid_sur_pre_no_nat)
        .sort_values('pid')
    )
    sur_pre_Rcur = (games
        .loc[:, games.columns.str.endswith('2')]
        .drop_duplicates()
        .rename(columns=lambda x: re.sub(r'(.*)2', r'\1', x))
    )
    pid_sur_pre_nat_Rcur = (pid_sur_pre_nat
        .merge(
            sur_pre_Rcur,
            how='outer', validate='one_to_one'
        )
    )
    return pid_sur_pre_nat_Rcur

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
        .rename(columns=lambda x: re.sub(r'(rating)(\d)', r'\1_T\2', x))
        .loc[:, [
            'event_place', 'event_date', 'event_significance',
            'pid1', 'surname1', 'prename1', 'nationality1', 'Rcur1',
            'pid2', 'surname2', 'prename2', 'nationality2', 'Rcur2',
            'score', 'PDcur', 'dRcur'
        ]]
    )

def append_games_anonymous(games: pd.DataFrame) -> pd.DataFrame:
    return (games
        .append(games
            .query('surname2 == "" & prename2 == ""')
            .loc[:, [
                'event_place', 'event_date', 'event_significance',
                'pid2', 'surname2', 'prename2', 'nationality2', 'Rcur2',
                'pid1', 'surname1', 'prename1', 'nationality1', 'Rcur1',
                'score', 'PDcur', 'dRcur'
            ]]
            .rename(columns=lambda x: re.sub(r'(.+)1', r'\g<1>0', x))
            .rename(columns=lambda x: re.sub(r'(.+)2', r'\g<1>1', x))
            .rename(columns=lambda x: re.sub(r'(.+)0', r'\g<1>2', x))
            .assign(
                score = lambda x: 1.0 - x.score,
                PDcur = lambda x: 1.0 - x.PDcur,
                dRcur = lambda x: -x.dRcur
            )
        )
        .sort_values(
            by=['pid1', 'event_date', 'Rcur2'],
            ascending=[True, False, False]
        )
        .reset_index(drop=True)
    )

def merge_standings_einf_pinf(standings: pd.DataFrame, events: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    return (events
        .loc[:, ['eid', 'gid', 'place', 'date']]
        .merge(standings, how='left', validate='one_to_many')
        .merge(players
            .drop(columns=['Rcur']),
            how='left', validate='many_to_one'
        )
        .loc[:, [
            'eid', 'gid', 'place', 'date', 'rank',
            'pid', 'surname', 'prename', 'nationality',
            'Rold', 'Rchg', 'Rnew', 'Reff_games',
            'standings_score', 'standings_buchholz', 'standings_median', 'standings_compa'
        ]]
    )

def merge_results_einf_pinf(results: pd.DataFrame, events: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    return (events
        .loc[:, ['eid', 'gid', 'place', 'date']]
        .merge(results, how='left', validate='one_to_many')
        .merge(standings
            .loc[:, [
                'eid', 'gid', 'place', 'date',
                'rank', 'pid', 'surname', 'prename', 'nationality', 'Rold', 'Rnew'
            ]]
            .rename(columns={x: x + '1' for x in ['rank', 'pid', 'surname', 'prename', 'nationality', 'Rold', 'Rnew']}),
            how='left', validate='many_to_one'
        )
        .merge(standings
            .loc[:, [
                'eid', 'gid', 'place', 'date',
                'rank', 'pid', 'surname', 'prename', 'nationality', 'Rold', 'Rnew'
            ]]
            .rename(columns={x: x + '2' for x in ['rank', 'pid', 'surname', 'prename', 'nationality', 'Rold', 'Rnew']}),
            how='outer', validate='many_to_one', indicator=True
        )
        .query('_merge != "right_only"')
        .drop(columns=['_merge'])
        .sort_values(by=['eid', 'gid', 'round', 'rank1'])
        .fillna({'pid2': 0})
        .astype(dtype={column: int for column in ['round', 'rank1', 'pid1', 'pid2']})
        .loc[:, [
            'eid', 'gid', 'place', 'date', 'round',
            'rank1', 'pid1', 'surname1', 'prename1', 'nationality1', 'Rold1', 'Rnew1',
            'rank2', 'pid2', 'surname2', 'prename2', 'nationality2', 'Rold2', 'Rnew2',
            'result'
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
    # Complete Kleier archive as of 2019-11-02
    # get_events.main(664)
    # get_players.main(2962)

    events      = kleier.load_dataset('events')
    standings   = kleier.load_dataset('standings').pipe(get_events.format_standings)
    results     = kleier.load_dataset('results').pipe(get_events.format_results)
    players     = kleier.load_dataset('players')
    games       = kleier.load_dataset('games').pipe(get_players.format_games)

    players     = merge_players_pinf(players, standings, games)
    games       = merge_games_pinf(games, players)
    games       = append_games_anonymous(games)
    standings   = merge_standings_einf_pinf(standings, events, players)
    results     = merge_results_einf_pinf(results, events, standings)

    event_cross  = event_cross_add_outcome(event_cross)
    event_table  = event_table_add_outcome(event_table, event_cross)

    event_cross  = event_cross_add_points(event_cross, event_index)
    event_table  = event_table_add_points(event_table, event_cross)

