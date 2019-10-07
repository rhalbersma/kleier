#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import pandas as pd

import kleier.utils

import player, tourn_table

def opponents(games: pd.DataFrame) -> pd.DataFrame:
    players1 = games[['pid1', 'name1']].drop_duplicates()
    players2 = games[['sur2', 'pre2', 'rating2']].drop_duplicates()
    players2['name2'] = players2.apply(lambda x: ' '.join([x['pre2'], x['sur2']]), axis=1)
    players = players1.merge(players2, 'outer', left_on='name1', right_on='name2')
    players.drop(['name1', 'name2'], axis=1, inplace=True)
    players.rename(columns=lambda x: re.sub(r'(.*)\d', r'\1', x), inplace=True)
    players.fillna({column: 0 for column in ['pid']}, inplace=True)
    players = players.astype(dtype={'pid': int})
    players.sort_values(by=['pid'], inplace=True)
    players.reset_index(drop=True, inplace=True)
    return players

def players_results(games: pd.DataFrame) -> tuple:
    players = unique_players(games)
    players1 = players.rename(columns=lambda x: x + '1')
    players2 = players.rename(columns=lambda x: x + '2')
    results = games.merge(players1).merge(players2)
    results.drop(['name1'], axis=1, inplace=True)
    columns = results.columns.to_list()
    columns = [columns[0]] + columns[-4:-1] + columns[1:4] + [columns[-1]] + columns[4:-4]
    results = results[columns]

    anonymous = results.query('pid2 == 0')
    columns = anonymous.columns.to_list()
    columns = columns[7:11] + columns[4:7] + columns[:4] + columns[11:]
    anonymous = anonymous[columns]
    anonymous['expected'] = 1 - anonymous['expected']
    anonymous['observed'] = 1 - anonymous['observed']
    anonymous['net_yield'] = -anonymous['net_yield']
    anonymous.rename(columns={
        'pid2': 'pid1', 'sur2': 'sur1', 'pre2': 'pre1', 'rating2': 'rating1',
        'pid1': 'pid2', 'sur1': 'sur2', 'pre1': 'pre2', 'rating1': 'rating2'
    }, inplace=True)

    results = pd.concat([anonymous, results])
    results.sort_values(by=['pid1', 'date', 'rating2', 'sur2', 'pre2'], ascending=[True, True, False, False, False], inplace=True)
    results.reset_index(drop=True, inplace=True)

    return players, results

    event_info['num_rounds'] = cross_table.groupby(['eid', 'gid'])['round'].max().reset_index(drop=True)
    event_info['num_players'] = standings.groupby(['eid', 'gid'])['rank'].max().reset_index(drop=True)
    event_info['num_games'] = len(cross_table.query('rank2 != 0').index)



def main():
    # player.download()
    player_info = kleier.load_dataset('player_info')
    game_balance = kleier.load_dataset('game_balance')

    # tourn_table.download()
    event_info = kleier.load_dataset('event_info')
    standings = kleier.load_dataset('standings')
    cross_table = kleier.load_dataset('cross_table')
