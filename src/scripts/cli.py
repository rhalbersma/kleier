#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import os

import click
import pandas as pd

from scripts import detect
from scripts import _fetch
from scripts import _parse

@click.group()
def kleier():
    pass

@kleier.command()
@click.confirmation_option(
    '-y', '--yes',
    hidden=True,
    prompt=
"""The database consists of 3600 files and takes 15 minutes to download.
After this operation, 160 Mb of additional disk space will be used. 
Do you want to continue?"""
)
@click.option(
    '-H', '--html-path',
    type=click.Path(writable=True),
    default='data/html',
    show_default=True,
    help='PATH is the directory where all .html files will be saved to.'
)
def fetch(html_path) -> None:
    """
    Fetch all Classic Stratego data from https://www.kleier.net/.
    """
    click.echo('Fetching the list of tournaments.')
    _fetch._tournaments_byplace(html_path)
    click.echo('Detecting the number of tournaments: ', nl=False)
    eid_seq = detect._tourn_tables(html_path)
    click.echo(f'{len(eid_seq)}')
    assert min(eid_seq) == 1
    assert max(eid_seq) == len(eid_seq)
    with click.progressbar(eid_seq, label=f'Fetching {len(eid_seq)} tournament tables:') as bar:
        for eid in bar:
            _fetch._tourn_table(eid, html_path)
    click.echo('Detecting the number of players: ', nl=False)
    pid_seq = range(1, 1 + max(detect._players(html_path)))
    click.echo(f'{len(pid_seq)}')
    with click.progressbar(pid_seq, label=f'Fetching {len(pid_seq)} player histories:') as bar:
        for pid in bar:
            _fetch._player(pid, html_path)
    click.echo('Fetching the rating history.')
    _fetch._rat_table(html_path, games=0, ntourn=len(eid_seq), items=len(eid_seq)*len(pid_seq))

@kleier.command()
@click.option(
    '-H', '--html-path',
    type=click.Path(exists=True),
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
def parse(html_path, pkl_path) -> None:
    """
    Parse all Classic Stratego data into the Pandas DataFrame format.
    """
    click.echo('Parsing the list of tournaments.')
    tournaments_byplace = _parse._tournaments_byplace(html_path)
    click.echo('Parsing the tournament events, groups, standings and results.')
    events, groups, standings, results = _parse._tourn_tables(html_path)
    click.echo('Parsing the player names and games.')
    names, games = _parse._players(html_path)
    click.echo('Parsing the rating history.')
    rat_table = _parse._rat_table(html_path)
    data = {
        'tournaments_byplace'   : tournaments_byplace,
        'events'                : events,
        'groups'                : groups,
        'standings'             : standings,
        'results'               : results,
        'names'                 : names,
        'games'                 : games,
        'rat_table'             : rat_table
    }
    os.makedirs(pkl_path, exist_ok=True)
    for key, value in data.items():
        value.to_pickle(os.path.join(pkl_path, key))
