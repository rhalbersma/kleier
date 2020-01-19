#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import os

import click
import pandas as pd

from . import concat
from . import detect
from . import get
from . import parse
from . import read

@click.group()
def kleier():
    pass

@kleier.command()
@click.confirmation_option(
    '-y', '--yes',
    hidden=True,
    prompt='The database takes ~15 minutes to download, contains ~3600 files, totalling ~160 Mb on disk. Do you want to continue?'
)
@click.option(
    '-H', '--html-path',
    type=click.Path(writable=True),
    default='data/html',
    show_default=True,
    help='PATH is the directory where all .html files will be saved to.'
)
def download(html_path) -> None:
    """
    Get all Classic Stratego data from https://www.kleier.net/.
    """
    click.echo('Downloading the list of tournaments.')
    get._tournaments_byplace(html_path)
    click.echo('Detecting the number of tournaments: ', nl=False)
    eid_seq = detect._tourn_tables(html_path)
    click.echo(f'{len(eid_seq)}')
    assert min(eid_seq) == 1
    assert max(eid_seq) == len(eid_seq)
    with click.progressbar(eid_seq, label=f'Downloading {len(eid_seq)} tournament tables:') as bar:
        for eid in bar:
            get._tourn_table(eid, html_path)
    click.echo('Detecting the number of players: ', nl=False)
    pid_seq = range(1, 1 + max(detect._players(html_path)))
    click.echo(f'{len(pid_seq)}')
    with click.progressbar(pid_seq, label=f'Downloading {len(pid_seq)} player histories:') as bar:
        for pid in bar:
            get._player(pid, html_path)
    click.echo('Downloading the rating history.')
    get._rat_table(html_path, games=0, ntourn=len(eid_seq), items=len(eid_seq)*len(pid_seq))

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
def pickle(html_path, pkl_path) -> None:
    """
    Pickle all Classic Stratego data into the Pandas DataFrame format.
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
