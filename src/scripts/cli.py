#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import os

import click
import pandas as pd

from . import concat
from . import parse
from . import read
from . import wget

@click.group()
def kleier():
    pass

@kleier.command()
@click.option(
    '-y', '--yes',
    hidden=True,
    is_flag=True,
    expose_value=False,
    prompt='Downloading the entire database takes about 13 minutes and consumes about 160 Mb on disk. Are you sure?'
)
@click.option(
    '-H', '--html-path',
    type=click.Path(writable=True),
    default='data/html',
    show_default=True,
    help='PATH is the directory where all .html files will be saved to.'
)
@click.argument('max_eid', type=int)
@click.argument('max_pid', type=int)
def download(max_eid, max_pid, html_path) -> None:
    """
    Download all Classic Stratego data from https://www.kleier.net/.
    """
    for pid in range(1, 1 + max_pid):
        wget._player(pid, html_path)
    wget._rat_table(html_path, games=0, ntourn=max_eid, items=max_eid*max_pid)
    for eid in range(1, 1 + max_eid):
        wget._tourn_table(eid, html_path)
    wget._tournaments_byplace(html_path)

@kleier.command()
@click.option(
    '-H', '--html-path',
    type=click.Path(writable=True),
    default='data/html',
    show_default=True,
    help='PATH is the directory where all .html files will be read from.'
)
@click.option(
    '-P', '--pkl-path',
    type=click.Path(writable=True),
    default='data/pkl',
    show_default=True,
    help='PATH is the directory where all .pkl files will be saved to.'
)
def pickle(html_path, pkl_path) -> None:
    """
    Pickle all Classic Stratego data in the Pandas DataFrame format.
    """
    names, games = concat._players(html_path)
    rat_table = parse._rat_table(read._rat_table(html_path))
    events, groups, standings, results = concat._tourn_tables(html_path)
    tournaments_byplace = parse._tournaments_byplace(read._tournaments_byplace(html_path))
    data = {
        'names'                 : names,
        'games'                 : games,
        'rat_table'             : rat_table,
        'events'                : events,
        'groups'                : groups,
        'standings'             : standings,
        'results'               : results,
        'tournaments_byplace'   : tournaments_byplace
    }
    os.makedirs(pkl_path, exist_ok=True)
    for key, value in data.items():
        value.to_pickle(os.path.join(pkl_path, key))
