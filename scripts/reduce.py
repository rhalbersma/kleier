#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import pandas as pd

def _significance(dates: pd.Series, base_date=None) -> pd.Series:
    # https://www.kleier.net/txt/rating_23.html#SEC23
    decay = 2.5731
    tropical_year = 365.246
    if base_date is None:
        base_date = dates.max()
    return np.round(np.exp(-((base_date - dates).dt.days / (decay * tropical_year))**2), 6)

def reduce_event_date_to_significance(events: pd.DataFrame, lists: pd.DataFrame, games: pd.DataFrame) -> bool:
    df0 = (events
        .loc[:, ['id', 'date']]
        .rename(columns={'id': 'event_id'})
        .assign(significance = lambda x: _significance(x.date))
    )
    df1 = lists
    df2 = (games
        .loc[:, ['event_id', 'significance']]
        .sort_values('event_id')
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return (
        np.isclose(df0['significance'], df1['significance'], rtol=1e-4).all() and
        np.isclose(df0['significance'], df2['significance'], rtol=1e-4).all() and
        np.isclose(df1['significance'], df2['significance']           ).all()
    )

def reduce_results_to_games(results: pd.DataFrame, games: pd.DataFrame) -> bool:
    df0 = (results
        .query('player_id_2 != 0')
        .merge(groups
            .loc[:, ['id', 'event_id']]
            .rename(columns={'id': 'group_id'})
            , how='left', on=['group_id'], validate='many_to_one'
        )
        .loc[:, ['event_id', 'player_id_1', 'player_id_2', 'W', 'unplayed']]
        .sort_values(['event_id', 'player_id_1', 'player_id_2', 'W', 'unplayed'])
        .reset_index(drop=True)
    )
    df1 = (games
        .loc[:, ['event_id', 'player_id_1', 'player_id_2', 'W', 'unplayed']]
        .sort_values(['event_id', 'player_id_1', 'player_id_2', 'W', 'unplayed'])
        .reset_index(drop=True)
    )
    return (
        df0.drop(columns=['unplayed']).equals(df1.drop(columns=['unplayed'])) and
        ((df0['unplayed'] == df1['unplayed']) | df1['unplayed'].isnull()).all()
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

def main() -> None:
    assert reduce_event_date_to_significance(events, lists, games)
    assert reduce_results_to_games(results, games)
