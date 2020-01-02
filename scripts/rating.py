#!/usr/bin/env python


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

