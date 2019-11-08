#!/usr/bin/env python

#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import math

import numpy as np
import pandas as pd
import scipy.stats as ss

import kleier.utils

import get_events
import get_players

def player_index_merge_pinf(player_index: pd.DataFrame, event_table: pd.DataFrame, player_cross: pd.DataFrame) -> pd.DataFrame:
    pid_name = player_index
    sur_pre_nat_name = (event_table
        .loc[:, ['sur', 'pre', 'nat']]
        .drop_duplicates()
        .assign(name = lambda x: x.pre + ' ' + x.sur))
    pid_sur_pre_nat = pd.merge(
        pid_name, sur_pre_nat_name,
        how='outer', indicator=True, validate='one_to_one')
    pid_sur_pre_no_nat = (pid_sur_pre_nat
        .query('_merge != "both"')
        .assign(
            sur = lambda x: x.name.str.split(expand=True)[1],
            pre = lambda x: x.name.str.split(expand=True)[0])
        .drop(columns=['name', '_merge']))
    pid_sur_pre_nat = (pid_sur_pre_nat
        .query('_merge == "both"')
        .drop(columns=['name', '_merge'])
        .append(pid_sur_pre_no_nat)
        .sort_values('pid'))
    name = (player_cross
        .loc[:, ['name1']]
        .rename(columns=lambda x: x[:-1])
        .drop_duplicates())
    sur_pre_rating_name = (player_cross
        .loc[:, ['sur2', 'pre2', 'rating2']]
        .rename(columns=lambda x: x[:-1])
        .drop_duplicates()
        .assign(name = lambda x: x.pre + ' ' + x.sur))
    sur_pre_rating = (pd.merge(
        name, sur_pre_rating_name,
        how='outer', validate='one_to_one')
        .drop(columns=['name']))
    pid_sur_pre_nat_rating = pd.merge(
        pid_sur_pre_nat, sur_pre_rating,
        how='outer', validate='one_to_one')
    return pid_sur_pre_nat_rating

def event_table_merge_einf_pinf(event_table: pd.DataFrame, event_index: pd.DataFrame, player_index: pd.DataFrame) -> pd.DataFrame:
    eid_gid_einf = (event_index
        .loc[:, ['eid', 'gid', 'place', 'date']])
    eid_gid_pinf = event_table
    eid_gid_einf_pinf = (pd
        .merge(eid_gid_einf, eid_gid_pinf))
    pid_pinf = (player_index
        .drop(columns=['rating']))
    eid_gid_einf_pid_pinf = (pd
        .merge(eid_gid_einf_pinf, pid_pinf, how='left'))
    columns = eid_gid_einf_pid_pinf.columns.to_list()
    columns = columns[:5] + [columns[-1]] + columns[5:-1]
    return eid_gid_einf_pid_pinf[columns]

def event_cross_merge_einf_pinf(event_cross: pd.DataFrame, event_index: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    eid_gid_einf = (event_index
        .loc[:, ['eid', 'gid', 'place', 'date']])
    eid_gid_rank_rinf = event_cross
    eid_gid_einf_rank_rinf = (pd
        .merge(eid_gid_einf, eid_gid_rank_rinf))
    eid_cols = ['eid', 'gid']
    pid_cols = ['rank', 'pid', 'sur', 'pre', 'nat', 'rating']
    eid_gid_rank_pid_pinf = (event_table
        .loc[:, eid_cols + pid_cols])
    eid_gid_rank1_pid1_pinf1 = eid_gid_rank_pid_pinf.rename(columns={c: c + '1' for c in pid_cols})
    eid_gid_rank2_pid2_pinf2 = eid_gid_rank_pid_pinf.rename(columns={c: c + '2' for c in pid_cols})
    eid_gid_einf_rank12_pid12_pinf12 = (eid_gid_einf_rank_rinf
        .merge(eid_gid_rank1_pid1_pinf1, how='left')
        .merge(eid_gid_rank2_pid2_pinf2, how='left')
        .astype(dtype={'pid2': pd.Int64Dtype()}))
    columns = eid_gid_einf_rank12_pid12_pinf12.columns.to_list()
    columns = columns[:6] + columns[8:13] + [columns[6]] + columns[13:] + [columns[7]]
    return eid_gid_einf_rank12_pid12_pinf12[columns]

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
    get_events.main(664)
    get_players.main(2962)

    events      = kleier.load_dataset('events')
    standings   = kleier.load_dataset('standings')
    results     = kleier.load_dataset('results')
    players     = kleier.load_dataset('players')
    games       = kleier.load_dataset('games')

    player_index = player_index_merge_pinf(player_index, event_table, player_cross)

    event_table  = event_table_merge_einf_pinf(event_table, event_index, player_index)
    event_cross  = event_cross_merge_einf_pinf(event_cross, event_index, event_table)

    event_cross  = event_cross_add_outcome(event_cross)
    event_table  = event_table_add_outcome(event_table, event_cross)

    event_cross  = event_cross_add_points(event_cross, event_index)
    event_table  = event_table_add_points(event_table, event_cross)

