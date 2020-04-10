#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import os

import click
import pandas as pd

from scripts._extract import _fetch
from scripts._extract import _scan
from scripts._transform import _format
from scripts._transform import _normalize
from scripts._transform import _parse
from scripts._transform import _reduce

dataset_names = [
    'tournaments',
    'events',
    'groups',
    'activity',
    'standings',
    'results',
    'names',
    'expected',
    'dates',
    'ratings',
    'history'
]

def _do_extract(html_path: str) -> None:
    click.echo('Fetching the list of tournaments.')
    _fetch._tournaments(html_path)
    click.echo('Scanning the number of tournaments: ', nl=False)
    eid_seq = _scan._tourn_tables(html_path)
    click.echo(f'{len(eid_seq)}')
    assert min(eid_seq) == 1
    assert max(eid_seq) == len(eid_seq)
    with click.progressbar(eid_seq, label=f'Fetching {len(eid_seq)} tournament tables:') as bar:
        for eid in bar:
            _fetch._tourn_table(eid, html_path)
    click.echo('Scanning the number of players: ', nl=False)
    pid_seq = range(1, 1 + max(_scan._players(html_path)))
    click.echo(f'{len(pid_seq)}')
    with click.progressbar(pid_seq, label=f'Fetching {len(pid_seq)} player histories:') as bar:
        for pid in bar:
            _fetch._player(pid, html_path)
    click.echo('Fetching the rating history.')
    _fetch._rat_table(html_path, games=0, ntourn=len(eid_seq), items=len(eid_seq)*len(pid_seq))

def _do_parse(html_path: str, pkl_path: str) -> None:
    click.echo('Parsing the list of tournaments.')
    tournaments = _parse._tournaments(html_path)
    click.echo('Parsing the tournament events, groups, activity, standings and results.')
    events, groups, activity, standings, results = _parse._tourn_tables(html_path)
    click.echo('Parsing the player names and expected results.')
    names, expected = _parse._players(html_path)
    click.echo('Parsing the rating history.')
    dates, ratings, history = _parse._rat_table(html_path)
    datasets = [
        tournaments,
        events,
        groups,
        activity,
        standings,
        results,
        names,
        expected,
        dates,
        ratings,
        history
    ]
    os.makedirs(pkl_path, exist_ok=True)
    for key, value in zip(dataset_names, datasets):
        value.to_pickle(os.path.join(pkl_path, key + '.pkl'))

def _do_format(pkl_path: str) -> None:
    assert os.path.exists(pkl_path)
    tournaments, events, groups, activity, standings, results, names, expected, dates, ratings, history = tuple(
        pd.read_pickle(os.path.join(pkl_path, file))
        for file in dataset_names
    )
    click.echo('Formatting the list of tournaments.')
    tournaments = _format._tournaments(tournaments)
    click.echo('Formatting the tournament events, groups, activity, standings and results.')
    events      = _format._events(events)
    groups      = _format._groups(groups)
    activity    = _format._activity(activity)
    standings   = _format._standings(standings)
    results     = _format._results(results)
    click.echo('Formatting the player names and games.')
    names       = _format._names(names)
    expected    = _format._expected(expected)
    click.echo('Formatting the rating history.')
    dates       = _format._dates(dates)
    ratings     = _format._ratings(ratings)
    history     = _format._history(history)
    datasets    = [
        tournaments,
        events,
        groups,
        activity,
        standings,
        results,
        names,
        expected,
        dates,
        ratings,
        history
    ]
    os.makedirs(pkl_path, exist_ok=True)
    for key, value in zip(dataset_names, datasets):
        value.to_pickle(os.path.join(pkl_path, key + '.pkl'))

def _do_normalize(pkl_path: str) -> None:
    assert os.path.exists(pkl_path)
    tournaments, events, groups, activity, standings, results, names, expected, dates, ratings, history = tuple(
        pd.read_pickle(os.path.join(pkl_path, file))
        for file in dataset_names
    )
    tournaments = _normalize._tournaments(tournaments)
    events      = _normalize._events(events)
    groups      = _normalize._groups(groups)
    names       = _normalize._names(names, standings)
    dates       = _normalize._dates(dates)
    ratings     = _normalize._ratings(ratings, names)
    history     = _normalize._history(history, events, names)
    activity    = _normalize._activity(activity, names)
    standings   = _normalize._standings(standings, names)
    results     = _normalize._results(results, standings)
    expected    = _normalize._expected(expected, events, names, dates, ratings, results)
    os.makedirs(pkl_path, exist_ok=True)
    datasets = [
        tournaments,
        events,
        groups,
        activity,
        standings,
        results,
        names,
        expected,
        dates,
        ratings,
        history
    ]
    for key, value in zip(dataset_names, datasets):
        value.to_pickle(os.path.join(pkl_path, key + '.pkl'))

def _do_transform(html_path: str, pkl_path: str) -> None:
    _do_parse(html_path, pkl_path)
    _do_format(pkl_path)
    _do_normalize(pkl_path)

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
def extract(html_path) -> None:
    """
    Extract all Classic Stratego data from https://www.kleier.net/.
    """
    _do_extract(html_path)

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
def transform(html_path, pkl_path) -> None:
    """
    Transform all Classic Stratego data into a normalized RDBS.
    """
    _do_transform(html_path, pkl_path)
