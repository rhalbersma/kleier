#!/usr/bin/env python

#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

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

