#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import pandas as pd

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
