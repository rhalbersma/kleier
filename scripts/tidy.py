#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import pandas as pd

import kleier.utils

import player, tourn_table

def download():
    player.download()
    tourn_table.download()

def load_datasets() -> tuple:
    player_info  = kleier.load_dataset('player_info')
    game_balance = kleier.load_dataset('game_balance')
    event_info   = kleier.load_dataset('event_info')
    standings    = kleier.load_dataset('standings')
    cross_table  = kleier.load_dataset('cross_table')
    return player_info, game_balance, event_info, standings, cross_table

def create_players(player_info: pd.DataFrame, standings: pd.DataFrame, game_balance: pd.DataFrame) -> tuple:
    p0 = player_info
    p1 = (standings
        .loc[:, ['sur', 'pre', 'nationality']]
        .drop_duplicates()
        .assign(name = lambda x: x.pre + ' ' + x.sur))
    p2 = pd.merge(p0, p1, how='outer', on=['name'], indicator=True, validate='one_to_one')
    p3 = (p2
        .query('_merge == "both"')
        .drop(columns=['name', '_merge'])
        .reset_index(drop=True))
    no_standings = (p2
        .query('_merge != "both"')
        .assign(sur = lambda x: x.name.str.split(expand=True)[1])
        .assign(pre = lambda x: x.name.str.split(expand=True)[0])
        .assign(rating = pd.np.nan)
        .drop(columns=['name', '_merge'])
        .reset_index(drop=True))

    p4 = (game_balance
        .loc[:, ['name1']]
        .rename(columns=lambda x: x[:-1])
        .drop_duplicates())
    p5 = (game_balance
        .loc[:, ['sur2', 'pre2', 'rating2']]
        .rename(columns=lambda x: x[:-1])
        .drop_duplicates() 
        .assign(name = lambda x: x.pre + ' ' + x.sur))
    p6 = (pd.merge(p4, p5, how='outer', on=['name'], indicator=True, validate='one_to_one')
        .drop(columns=['name']))        
    p7 = p6.drop(columns=['_merge'])

    p8 = pd.merge(p3, p7, how='outer', on=['sur', 'pre'], indicator=True, validate='one_to_one')
    players = (p8
        .drop(columns=['_merge'])
        .append(no_standings)
        .sort_values(by=['pid'])
        .reset_index(drop=True))
    no_results = (p8
        .query('_merge != "both"')
        .drop(columns=['_merge'])
        .reset_index(drop=True))

    p9 = (p6
        .query('_merge != "both"')
        .drop(columns=['_merge']))
    no_name = pd.merge(p3, p9, how='right', on=['sur', 'pre'], validate='one_to_one')

    return players, no_name, no_results, no_standings

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

    event_info['rounds'] = cross_table.groupby(['eid', 'gid'])['round'].max().reset_index(drop=True)
    event_info['players'] = standings.groupby(['eid', 'gid'])['rank'].max().reset_index(drop=True)
    event_info['games'] = cross_table.query('rank2 != 0').groupby(['eid', 'gid']).size().reset_index(drop=True)


def main():
    pass
