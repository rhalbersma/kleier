#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import os
import re

import pandas as pd
from typing import Tuple

from . import parse
from . import read

def _players(path: str) -> Tuple[pd.DataFrame]:
    pid_seq = [
        int(os.path.splitext(file)[0].split('-')[-1])
        for file in os.listdir(path)
        if re.match(r'player-\d+\.html', file)
    ]
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            parse._player(pid, read._player(pid, path))
            for pid in pid_seq
        ])
    )

def _tourn_tables(path: str) -> Tuple[pd.DataFrame]:
    eid_seq = [
        int(os.path.splitext(file)[0].split('-')[-1])
        for file in os.listdir(path)
        if re.match(r'tourn_table-\d+\.html', file)
    ]
    return tuple(
        pd.concat(list(t), ignore_index=True, sort=False)
        for t in zip(*[
            parse._tourn_table(eid, read._tourn_table(eid, path))
            for eid in eid_seq
        ])
    )
