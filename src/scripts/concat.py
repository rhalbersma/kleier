#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import pandas as pd
from typing import Sequence, Tuple

from . import detect
from . import parse
from . import read

def _players(path: str) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            parse._player(pid, read._player(pid, path))
            for pid in detect._files(r'player-\d+\.html', path)
        ])
    )

def _tourn_tables(path: str) -> Tuple[pd.DataFrame]:
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            parse._tourn_table(eid, read._tourn_table(eid, path))
            for eid in detect._files(r'tourn_table-\d+\.html', path)
        ])
    )
