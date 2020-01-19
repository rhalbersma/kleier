#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import os
import re

from typing import Sequence

from . import read

def _files(regex: str, path: str) -> Sequence[int]:
    return list(sorted({
        int(os.path.splitext(file)[0].split('-')[-1])
        for file in os.listdir(path)
        if re.match(regex, file)
    }))

def _players(html_path: str) -> Sequence[int]:
    return list(sorted(set.union(*[{
            int(td.find('a')['href'].split('=')[-1])
            for td in read._tourn_table(eid, html_path).find_all('td', {'class': 'name'})
            if td.text
        }
        for eid in _files(r'tourn_table-\d+\.html', html_path)
    ])))

def _tourn_tables(html_path: str) -> Sequence[int]:
    soup = read._tournaments_byplace(html_path)
    return list(sorted({
        int(eid.get('href').split('=')[1])
        for nat in soup.find('ul', {'class': 'nat'}).find_all('li', recursive=False)
        for eid in nat.find_all('a')
    }))
