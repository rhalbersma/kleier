#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import os
import pkg_resources

import pandas as pd

def get_resource(basename: str) -> str:
    return pkg_resources.resource_filename(__name__, os.path.join('data', basename))

def save_dataset(df: pd.DataFrame, name: str) -> None:
    df.to_pickle(get_resource(name + '.pkl'))

def load_dataset(name: str) -> pd.DataFrame:
    return pd.read_pickle(get_resource(name + '.pkl'))

def data_listdir() -> list:
    return [
        os.path.splitext(df)[0]
        for df in pkg_resources.resource_listdir(__name__, 'data')
    ]
