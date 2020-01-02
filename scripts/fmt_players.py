#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import pandas as pd

def format_players(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .sort_values('pid')
        .reset_index(drop=True)
        .rename(columns={
            'pid' : 'id'
        })
    )

def format_games(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .pipe(lambda x: x
            .set_axis(x
                .columns
                .to_flat_index()
                .map('_'.join)
                , axis='columns', inplace=False
            )
        )
        .rename(columns=lambda x: x.strip('_'))
        .rename(columns=lambda x: x.replace(u'\xa0\u2191', ''))
        .rename(columns=lambda x: x.replace(' ', '_'))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: re.sub(r'event_(.*)', r'\1', x))
        .rename(columns=lambda x: re.sub(r'opponent_(.*)', r'\g<1>_2', x))
        .rename(columns=lambda x: re.sub(r'result_(.*)', r'\1', x))
        .rename(columns=lambda x: re.sub(r'(.*)name(.*)', r'\1\2', x))
        .rename(columns={
            'pid'      : 'player_id_1',
            'rating_2' : 'R_2',         # R = player 2's current rating
            'expected' : 'We',          # We = the expected score W (Elo, 1978)
            'observed' : 'W',           # W = the number of wins, draws counting 1/2 (Elo, 1978)
            'net_yield': 'dW'           # dW = W - We ('d' from 'delta')
        })
        .astype(dtype={
            'date': 'datetime64[ns]',
            'R_2'  : pd.Int64Dtype()
        })
        .loc[:, [
            'date', 'place', 'player_id_1', 'sur_2', 'pre_2',
            'R_2', 'significance',
            'unplayed', 'W', 'We', 'dW'
        ]]
    )
