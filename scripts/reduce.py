#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import pandas as pd

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

def reduce_history_to_activity(history: pd.DataFrame, activity: pd.DataFrame) -> bool:
    df = pd.merge(activity, history, how='left', on=['player_id', 'event_id'], validate='one_to_one')
    return df.shape[0] == activity.shape[0]

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

    assert reduce_results_to_games(results, games)
    games = games.drop(columns=['unplayed', 'W', 'dW'])

    assert compare_tot_games_ratings_results(ratings, results)
    ratings = ratings.drop(columns=['tot_games'])

    assert reduce_history_to_activity(history, activity)
    activity = activity.drop(columns=['Rn', 'Ro', 'dR'])

