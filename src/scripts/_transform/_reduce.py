#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import pandas as pd

# helper functions for _games

def _significance(games: pd.DataFrame, lists: pd.DataFrame) -> bool:
    df0 = lists
    df1 = (games
        .loc[:, ['event_id', 'significance']]
        .drop_duplicates()
        .sort_values('event_id')
        .reset_index(drop=True)
    )
    return np.isclose(
        df0.significance, 
        df1.significance
    ).all()

def _ratings(games: pd.DataFrame, names: pd.DataFrame, ratings: pd.DataFrame) -> bool:
    df0 = (ratings
        .loc[:, ['player_id', 'R']]
        .sort_values('player_id')
        .reset_index(drop=True)
    )
    df1 = (names
        .rename(columns={'id': 'player_id_1'})
        .query('pre.notnull() & sur.notnull()')
        .loc[:, ['player_id_1']]
        .merge(games
            .loc[:, ['player_id_1', 'R_1']]
            .drop_duplicates()
            , how='left', on='player_id_1', validate='one_to_one'
        )        
        .query('R_1.notnull()')
        .reset_index(drop=True)
    )
    df2 = (names
        .rename(columns={'id': 'player_id_2'})
        .query('pre.notnull() & sur.notnull()')
        .loc[:, ['player_id_2']]
        .merge(games
            .loc[:, ['player_id_2', 'R_2']]
            .drop_duplicates()
            , how='left', on='player_id_2', validate='one_to_one'
        )        
        .query('R_2.notnull()')
        .reset_index(drop=True)
    )
    return df0.R.equals(df1.R_1) and df0.R.equals(df2.R_2)

def _unplayed_W(games: pd.DataFrame, results: pd.DataFrame, groups: pd.DataFrame) -> bool:
    df0 = (results
        .query('player_id_2 != 0')
        .merge(groups
            .loc[:, ['id', 'event_id']]
            .rename(columns={'id': 'group_id'})
            , how='left', on='group_id', validate='many_to_one'
        )
        .loc[:,      ['event_id', 'player_id_1', 'player_id_2', 'W', 'unplayed']]
        .sort_values(['event_id', 'player_id_1', 'player_id_2', 'W', 'unplayed'])
        .reset_index(drop=True)
    )
    df1 = (games
        .loc[:,      ['event_id', 'player_id_1', 'player_id_2', 'W', 'unplayed']]
        .sort_values(['event_id', 'player_id_1', 'player_id_2', 'W', 'unplayed'])
        .reset_index(drop=True)
    )
    return ((df0.unplayed == df1.unplayed) | df1.unplayed.isnull()).all() and df0.W.equals(df1.W)

def _games(games: pd.DataFrame, lists: pd.DataFrame, names: pd.DataFrame, ratings: pd.DataFrame, results: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    assert _significance(df, lists)
    df = df.drop(columns=['significance'])
    assert _ratings(df, names, ratings)
    df = df.drop(columns=['R_1', 'R_2'])
    assert _unplayed_W(df, results, groups)
    df = df.drop(columns=['unplayed', 'W', 'dW'])    
    df = (df
        .loc[:, ['player_id_1', 'player_id_2', 'We']]
        .drop_duplicates()
        .sort_values(['player_id_1', 'player_id_2'])
        .reset_index(drop=True)
    )
    assert not df.duplicated(subset=['player_id_1', 'player_id_2']).any()
    return df

def activity_has_redundant_ratings(activity: pd.DataFrame, history: pd.DataFrame) -> bool:
    return (activity
        .merge(history, how='left', validate='one_to_one')
        .equals(activity)
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

def main() -> None:
    assert reduce_events_date_to_significance(events, lists, games)
    games = games.drop(columns=['significance'])


    assert compare_tot_games_ratings_results(ratings, results)
    ratings = ratings.drop(columns=['tot_games'])

    assert reduce_history_to_activity(history, activity)
    activity = activity.drop(columns=['Rn', 'Ro', 'dR'])
