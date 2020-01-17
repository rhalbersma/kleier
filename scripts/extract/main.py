#!/usr/bin/env python

#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import sys

import pandas as pd
from typing import Tuple

from . import wget
from . import open
from . import parse

def _tourn_tables(prefix: str, max_eid: int) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            parse.tourn_table(open.tourn_table(prefix, eid), eid)
            for eid in range(1, 1 + max_eid)
        ])
    )

def _players(prefix: str, max_pid: int) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            parse.player(open.player(prefix, pid), pid)
            for pid in range(1, 1 + max_pid)
        ])
    )

def _tournaments_byplace(prefix: str) -> pd.DataFrame:
    return parse.tournaments_byplace(open.tournaments_byplace(prefix))

def _rat_table(prefix: str) -> pd.DataFrame:
    return parse.rat_table(open.rat_table(prefix))

def main(prefix: str, max_eid: int, max_pid: int) -> None:
    # extract.wget.main(prefix, max_eid, max_pid)
    events, groups, standings, results = _tourn_tables(prefix, max_eid)
    names, games = _players(prefix, max_pid)
    tournaments_byplace = _tournaments_byplace(prefix)
    rat_table = _rat_table(prefix)

if __name__ == '__main__':
    sys.exit(main(str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])))
