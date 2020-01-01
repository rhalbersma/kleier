#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import numpy as np
import pandas as pd

def _rating_lists(rat_table: pd.DataFrame) -> pd.DataFrame:
    return (rat_table
        .filter(regex='Rating')
        .columns
        .to_frame()
        .reset_index(drop=True)
        .reset_index()
        .drop(columns=[0])
        .rename(columns={
            'index': 'id',
            1: 'place',
            2: 'date',
            3: 'significance'
        })
        .astype(dtype={
            'date'        : 'datetime64[ns]',
            'significance': float
        })
        .loc[:, [
            'id', 'date', 'place', 'significance'
        ]]
    )

# TODO: least squares fit of Gaussian over y = [ 1.0, 0.8, 0.55, 0.3, 0.05 ] for x = [ 0, 1, 2, 3, 4 ]

def _significance(dates: pd.Series, base_date: 'datetime64[ns]') -> pd.Series:
    decay = 2.5731              # https://www.kleier.net/txt/rating_23.html#SEC23
    tropical_year = 365.246
    return np.round(np.exp(-((base_date - dates).dt.days / (decay * tropical_year))**2), 6)

def _rating_history(rat_table: pd.DataFrame, rating_lists: pd.DataFrame) -> pd.DataFrame:
    return (pd
        .wide_to_long(rat_table
            .pipe(lambda x: x
                .set_axis(pd.
                    Index(
                        [ t[2:] for t in x.drop(x.filter(regex='Rating'), axis='columns').columns ] +
                        [ ('Rating', str(id)) for id in rating_lists['id'] ]
                    )
                    .to_flat_index()
                    .map(dict.fromkeys)
                    .map(list)
                    .map('_'.join)
                    , axis='columns', inplace=False
                )
            )
            .assign(ranking_all = lambda x: x.index + 1)
            , ['Rating_'], i='ranking_all', j='list_id'
        )
        .reset_index()
        .merge(rating_lists.rename(columns={'id': 'list_id'}), on='list_id')
        .rename(columns=lambda x: x.strip('._'))
        .rename(columns=lambda x: x.lower())
        .rename(columns=lambda x: re.sub(r'ranking(_)(\w{3}).*', r'\2\1rank', x))
        .rename(columns=lambda x: re.sub(r'(games)(_)(\w{3}).*', r'\3\2\1', x))
        .rename(columns=lambda x: re.sub(r'(.*)name', r'\1', x))
        .rename(columns={'rating': 'R'})
        .assign(nat = lambda x: x.nat_rank.str.split('/').str[-1])
        .assign(nat_rank = lambda x:
            np.where(
                x.nat_rank.str[0].str.isdigit(),
                x.nat_rank.str.split('/').str[0],
                np.nan
            )
        )
        .assign(int_rank = lambda x:
            np.where(
                x.int_rank.str[0].str.isdigit(),
                x.int_rank,
                np.nan
            )
        )
        .loc[:, [
            'date', 'place', 'sur', 'pre', 'nat', 
            'R', 'all_rank', 'int_rank', 'nat_rank', 'eff_games', 'tot_games', 'significance'
        ]]
    )

def _rating_history(rat_table: pd.DataFrame) -> Tuple[pd.DataFrame]
    rating_events = _rating_events(rat_table, events)
    assert rating_events.equals(rating_events.sort_values('id', ascending=False))
    assert np.isclose(_significance(rating_events['date']), rating_events['significance'], rtol=1e-4).all()
    rating_history = _rating_history(rat_table)
    ratings = (rating_history
        .query('event_id == @max_eid')
        .rename(columns={'index': 'id'})
    )
    history = (rating_history
        .sort_values(['event_id', 'index'])
        .reset_index()
        .reset_index()
        .rename(columns={'index': 'id'})
        .loc[:, ['event_id', 'Prename', 'Surname', 'R_']]
    )
    return ratings, history

def format_ratings(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    return (df

        .rename(columns=lambda x: re.sub(r'r_(\d+)', r'R\1', x))

        .replace({'unrated': np.nan})
        .pipe(lambda x: x.astype(dtype={column: float           for column in x.columns if column.endswith('rank')}))
        .pipe(lambda x: x.astype(dtype={column: pd.Int64Dtype() for column in x.columns if column.endswith('rank')}))
        .pipe(lambda x: x.astype(dtype={column: float           for column in x.columns if column.startswith('R')}))
        .pipe(lambda x: x.astype(dtype={column: pd.Int64Dtype() for column in x.columns if column.startswith('R')}))
        .pipe(lambda x: x.loc[:, x.columns.to_list()[:6] + x.columns.to_list()[-1:] + x.columns.to_list()[6:-1]])
    )
