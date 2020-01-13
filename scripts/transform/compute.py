#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import pandas as pd

def date_significance(date: pd.Series, max_date: 'datetime64[ns]') -> pd.Series:
    # https://www.kleier.net/txt/rating_23.html#SEC23
    decay = 2.5731
    tropical_year = 365.246
    return np.round(np.exp(-((max_date - date).dt.days / (decay * tropical_year))**2), 6)

def significance_compute(events: pd.DataFrame) -> pd.Series:
    assert events.equals(events.sort_values('id'))
    date = events.date
    max_date = date.max()
    return date_significance(date, max_date)

def significance_extract_games(games: pd.DataFrame) -> pd.Series:
    return (games
        .loc[:, ['event_id', 'significance']]
        .drop_duplicates(subset='event_id')
        .sort_values('event_id')
        .reset_index(drop=True)
        .significance
    )

def significance_extract_lists(lists: pd.DataFrame) -> pd.Series:
    assert lists.equals(lists.sort_values('event_id'))
    return lists.significance

def significance_compare(events: pd.DataFrame, games: pd.DataFrame, lists: pd.DataFrame) -> bool:
    df = pd.concat([
            significance_compute(events).rename('compute'),
            significance_extract_games(lists).rename('extract_games'),
            significance_extract_lists(lists).rename('extract_lists')
        ],
        axis='columns'
    )
    return (
        np.isclose(df.compute, df.extract_games, rtol=1e-4).all() and
        np.isclose(df.compute, df.extract_lists, rtol=1e-4).all() and
        np.isclose(df.extract_games, df.extract_lists,    ).all()
    )

def tot_games_compute(results: pd.DataFrame) -> pd.DataFrame:
    return (results
        .query('not unplayed')
        .groupby('player_id_1')
        .size()
        .to_frame('tot_games')
        .reset_index()
        .rename(columns={'player_id_1': 'player_id'})
    )

def tot_games_extract(ratings: pd.DataFrame) -> pd.DataFrame:
    return (ratings
        .loc[:, ['player_id', 'tot_games']]
        .sort_values('player_id')
        .reset_index(drop=True)
    )

def tot_games_compare(results: pd.DataFrame, ratings: pd.DataFrame) -> bool:
    df = pd.merge(
        tot_games_compute(results),
        tot_games_extract(ratings),
        how='right', on='player_id', suffixes=('_compute', '_extract'), validate='one_to_one'
    )
    return (df.tot_games_compute == df.tot_games_extract).all()

def eff_games_compute(results: pd.DataFrame, groups: pd.DataFrame, lists: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    return (results
        .query('not unplayed')
        .merge(groups
            .loc[:, ['id', 'event_id']]
            .rename(columns={'id': 'group_id'})
            , how='left', on='group_id', validate='many_to_one'
        )
        .drop(columns=['group_id'])
        .merge(lists)
        .merge(history
            .groupby('player_id')
            .first()
            .reset_index()
            .loc[:, ['player_id', 'event_id']]
            .rename(columns={
                'player_id': 'player_id_1',
                'event_id' : 'rated_since'
            })
        )
        .assign(rated_event_id = lambda x: x[['event_id', 'rated_since']].max(axis=1))
        .merge(history
            .loc[:, ['event_id', 'player_id', 'Rn']]
            .rename(columns={
                'player_id': 'player_id_2',
                'event_id': 'rated_event_id'
            })
            , how='left'
        )
        .query('Rn.notnull()')
        .groupby('player_id_1')
        .apply(lambda p: p.assign(event_nr = lambda x: x.event_id.rank(method='dense')))
        .assign(smooth = lambda x: np.where(x.event_nr == 1.0, 0.2, np.where(x.event_nr == 2.0, 0.5, 1.0)))
        .assign(smooth_sig = lambda x: x.smooth * x.significance)
        .groupby('player_id_1')
        .agg(eff_games=('smooth_sig', 'sum'))
        .assign(eff_games = lambda x: np.round(x.eff_games, 3))
        .reset_index()
        .rename(columns={'player_id_1': 'player_id'})
    )

def eff_games_extract(ratings: pd.DataFrame) -> pd.DataFrame:
    return (ratings
        .loc[:, ['player_id', 'eff_games']]
        .sort_values('player_id')
        .reset_index(drop=True)
    )

def eff_games_compare(results: pd.DataFrame, groups: pd.DataFrame, lists: pd.DataFrame, history: pd.DataFrame, ratings: pd.DataFrame) -> bool:
    df = pd.merge(
        eff_games_compute(results, groups, lists, history),
        eff_games_extract(ratings),
        how='right', on='player_id', suffixes=('_compute', '_extract'), validate='one_to_one'
    )
    return (df.eff_games_extract == df.eff_games_compute).all()
