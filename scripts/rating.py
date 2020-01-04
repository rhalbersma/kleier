#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import pandas as pd
import scipy.stats as ss

def pd_norm(dR, sigma):
    return ss.norm.cdf(dR, 0, sigma)

sigma_0 = 200 * np.sqrt(2)      # standard deviation of dR if R_1 and R_2 have standard deviation 200
sigma_1 = 2000 / 7              # easy manual computation of dR * 7 / 2000 to 4 digits for table lookup

def pd_logistic(dR, s):
    return ss.logistic.cdf(dR, 0, s)

s_0 = 400 / np.log(10)          # converstion from base-10 and scale = 400
s_1 = 100 * np.sqrt(np.pi)      # same slope at dR = 0 as norm.pdf with sigma = 200 * sqrt(2)
s_2 = 200 * np.sqrt(6) / np.pi  # equal variance as norm.pdf with sigma = 200 * sqrt(2)

def reduce_prediction(games):
    df = (games
        .query('We.notnull()')
        .loc[:,['player_id_1', 'player_id_2', 'R_1', 'R_2', 'We']]
        .drop_duplicates()
        .sort_values(
            ['player_id_1', 'R_2'],
            ascending=[True, False]
        )
        .reset_index(drop=True)
        .astype(dtype={column: float for column in ['R_1', 'R_2']})
        .assign(
            pd_sigma_0 = lambda x: pd_norm(x.R_1 - x.R_2, sigma_0),
            pd_sigma_1 = lambda x: pd_norm(x.R_1 - x.R_2, sigma_1),
            pd_s_0 = lambda x: pd_logistic(x.R_1 - x.R_2, s_0),
            pd_s_1 = lambda x: pd_logistic(x.R_1 - x.R_2, s_1),
            pd_s_2 = lambda x: pd_logistic(x.R_1 - x.R_2, s_2)
        )
    )
    df.loc[:, [
        'We', 'pd_sigma_0', 'pd_sigma_1', 'pd_s_0', 'pd_s_1', 'pd_s_2'
    ]].corr()

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

