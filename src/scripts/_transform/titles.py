#          Copyright Rein Halbersma 2019-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import pandas as pd

from scripts._extract import _fetch

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

def min_rating(results: pd.DataFrame, names: pd.DataFrame, events: pd.DataFrame, activity: pd.DataFrame) -> pd.DataFrame:
    df = (results
        .merge(activity
            .loc[:, ['pid', 'eid', 'R']]
            .rename(columns={
                'pid': 'pid1',
                'R'  : 'R1'
            })
        )
        .merge(activity
            .loc[:, ['pid', 'eid', 'R']]
            .rename(columns={
                'pid': 'pid2',
                'R'  : 'R2'
            })
        )
        .assign(rated = lambda x: np.where(x.unplayed | x.R1.isnull() | x.R2.isnull(), 0, 1))
        .loc[:, ['pid1', 'eid', 'rated']]
        .rename(columns={'pid1': 'pid'})
        .groupby(['pid', 'eid'])
        .agg(rated = ('rated', 'sum'))
        .query('rated > 0')
        .groupby('pid')
        .agg(rated_games = ('rated', 'cumsum'))
        .reset_index()
        .merge(names, how='left')
        .merge(events, how='left')
        .merge(activity, how='left')
        .query('rated_games >= @min_title_games & eff_games >= @min_title_games // 2')
        .reset_index(drop=True)
        .drop(columns='dR')
        .pipe(lambda x: x.join(
            x.groupby('pid').agg(max_R = ('R', 'cummax'))
        ))
        .astype(dtype={'max_R': 'Int64'})
        .assign(
            CM = lambda x: x.max_R >= min_rating_CM,
            FM = lambda x: x.max_R >= min_rating_FM,
            IM = lambda x: x.max_R >= min_rating_IM,
            GM = lambda x: x.max_R >= min_rating_GM
        )
    )
    df.query('CM').groupby('pid').head(1).sort_values(['eid', 'pid'])
    df.query('FM').groupby('pid').head(1).sort_values(['eid', 'pid'])
    df.query('IM').groupby('pid').head(1).sort_values(['eid', 'pid'])
    df.query('GM').groupby('pid').head(1).sort_values(['eid', 'pid'])
