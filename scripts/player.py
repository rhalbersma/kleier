#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import re

import bs4
import pandas as pd
import requests

import kleier.utils

# We recode missing names as follows:
anonymous = [
    'Anonymous Anonymous'
]

# The following players appear to be miss-spellings:
miss_spelling = [
    'Andrey Okhrimets', # 'Andrey Ochrimets' or 'Andriy Okrynets'
    'Sofiya Shevchuk'   # 'Sophiya Shevchuk'
]

# The following player has played only byes and withdrawals:
only_byes_and_withdrawals = [
    'John Jones'
]

def _parse_table(pid, table) -> pd.DataFrame:
    df = pd.read_html(str(table), header=2)[0]
    df.rename(columns=lambda x: str.lower(x).replace(' ', '_'), inplace=True)
    df.rename(columns=lambda x: 'date' if str(x).startswith('date') else x, inplace=True)
    df.rename(columns={
        'surname': 'sur2',
        'prename': 'pre2',
        'rating' : 'rating2'
    }, inplace=True)

    # Mark missing names as "Anonymous".
    df.fillna({column: 'Anonymous' for column in ['sur2', 'pre2']}, inplace=True)

    header_rows = table.find('thead').find_all('tr')
    name = header_rows[0].find('th').text.split('Game Balance of ')[1].split(',')[0]
    df['pid1'] = pid
    df['name1'] = name

    columns = df.columns.to_list()
    columns = columns[-2:] + columns[:3] + columns[3:-2]
    df = df[columns]
    
    df = df.astype(dtype={column: 'datetime64[ns]' for column in ['date']})
    df = df.astype(dtype={column: float            for column in ['rating2']})
    df = df.astype(dtype={column: pd.Int64Dtype()  for column in ['rating2']})

    df.sort_values(by=['date', 'rating2'], ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def _player(pid) -> tuple:
    url = f'https://www.kleier.net/cgi/player.php?pid={pid}'
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    header = soup.find_all('h1')
    tables = soup.find_all('table')
    assert len(header) != 0 or len(tables) == 0
    name = 'Anonymous Anonymous' if len(header) == 0 else header[0].text.split('History of ')[1]
    player_info = pd.DataFrame(
        data   =[(pid,   name)],
        columns=['pid', 'name']
    )
    game_balance = None if len(tables) == 0 else _parse_table(pid, tables[0])
    assert len(tables) != 0 or name in anonymous + miss_spelling + only_byes_and_withdrawals
    return player_info, game_balance

def download(num_pid=2952):
    player_info, game_balance = tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            _player(pid)
            for pid in range(1, num_pid + 1)
        ])
    )
    kleier.utils._save_dataset(player_info, 'player_info')
    kleier.utils._save_dataset(game_balance, 'game_balance')
