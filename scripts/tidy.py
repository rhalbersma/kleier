#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import math

import numpy as np
import pandas as pd
import scipy.stats as ss

import kleier.utils

import player, event

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

def dp(pts, P, s = 200 * math.sqrt(2)):
    return np.round(np.where(P == 0, np.nan, s * np.where(
        (0 == pts) | (pts == P),
        ss.norm.ppf((pts + .5) / (P + 1)) * (P + 1) / P,
        ss.norm.ppf( pts       /  P     ))))

def poty(player_index: pd.DataFrame, event_cross: pd.DataFrame) -> pd.DataFrame:
    df = (event_cross
        .query('rank2 != 0')
        .assign(year = lambda x: x.date.dt.year)
        .fillna({'rating2': 600})
        .groupby(['year', 'pid1'])
        .agg(
            games = ('pts', 'count'),
            pts = ('pts', 'sum'),
            Ra = ('rating2', 'mean'))
        .reset_index()
        .rename(columns={'pid1': 'pid'})
        .round({'Ra': 0})
        .assign(
            p = lambda x: x.pts / x.games,
            dp = lambda x: dp(x.pts, x.games),
            Rp = lambda x: x.Ra + x.dp)
        .astype(dtype={'Ra': int, 'dp': int, 'Rp': int}))
    df2 = pd.merge(player_index, df)
    columns = ['year', 'sur', 'pre', 'nat', 'games', 'pts', 'p', 'dp', 'Ra', 'Rp']
    return df2[columns].sort_values(['year', 'Rp'], ascending=[True, False])

        .query('P != 0')
        .assign(
            dp = lambda x: dp(x.pts, x.P),
            Rp = lambda x: x.Ra + x.dp)
        .astype(dtype={'Ra': int, 'dp': int, 'Rp': int})
        .drop(columns=['change', 'eff_games', 'score', 'buchholz', 'median', 'compa']))

delta           = 800

# 1.22
min_p           = .35

# 1.3
min_title_games = 30

min_rating_FM   = 2300 - delta  # 1.31
min_rating_CM   = 2200 - delta  # 1.32
min_rating_WFM  = 2100 - delta  # 1.33
min_rating_WCM  = 2000 - delta  # 1.34

# 1.46b
adj_rating_GM   = 2200 - delta
adj_rating_IM   = 2050 - delta
adj_rating_WGM  = 2000 - delta
adj_rating_WIM  = 1850 - delta

# 1.46d
rating_floor    = 1000 - delta

# 1.48
norm_rating_GM  = 2600 - delta
norm_rating_IM  = 2450 - delta
norm_rating_WGM = 2400 - delta
norm_rating_WIM = 2250 - delta

# 1.48a
avg_rating_GM   = 2380 - delta
avg_rating_IM   = 2230 - delta
avg_rating_WGM  = 2180 - delta
avg_rating_WIM  = 2030 - delta

# 1.49
min_norm_games  = 27

# 1.53
min_rating_GM   = 2500 - delta
min_rating_IM   = 2400 - delta
min_rating_WGM  = 2300 - delta
min_rating_WIM  = 2200 - delta

def player_history(event_table: pd.DataFrame) -> pd.DataFrame:
    df = (event_table
        .loc[:, ['eid', 'gid', 'place', 'date', 'rank', 'pid', 'sur', 'pre', 'nat', 'rating', 'P']]
        .sort_values(['pid', 'eid', 'gid'])
        .reset_index(drop=True))
    df = (df
        .join(df
            .groupby('pid')
            .agg(games = ('P', 'cumsum')))
        .query('games >= @min_title_games'))
    df = (df
        .join(df
            .groupby('pid')
            .agg(max_rating = ('rating', 'cummax')))
        .assign(
            FM  = lambda x: x.max_rating >= min_rating_FM,
            CM  = lambda x: x.max_rating >= min_rating_CM)
        .astype(dtype={
            'FM' : int,
            'CM' : int})
        .drop(columns=['rating', 'P']))
    player_CM = (df
        .query('CM == 1')
        .drop(columns=['CM', 'FM'])
        .groupby('pid')
        .first()
        .sort_values(['eid', 'gid', 'rank'])
        .reset_index()
        .assign(title = 'CM'))
    player_FM = (df
        .query('FM == 1')
        .groupby('pid')
        .first()
        .sort_values(['eid', 'gid', 'rank'])
        .reset_index()
        .assign(title = 'FM'))

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
    #player.download()
    #event.download()

    player_index = kleier.load_dataset('player_index')
    player_cross = kleier.load_dataset('player_cross')
    event_index  = kleier.load_dataset('event_index')
    event_table  = kleier.load_dataset('event_table')
    event_cross  = kleier.load_dataset('event_cross')

    player_index = player_index_merge_pinf(player_index, event_table, player_cross)

    event_table  = event_table_merge_einf_pinf(event_table, event_index, player_index)
    event_cross  = event_cross_merge_einf_pinf(event_cross, event_index, event_table)

    event_cross  = event_cross_add_outcome(event_cross)
    event_table  = event_table_add_outcome(event_table, event_cross)

    event_cross  = event_cross_add_points(event_cross, event_index)
    event_table  = event_table_add_points(event_table, event_cross)

