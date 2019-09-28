#!/usr/bin/env python

#          Copyright Rein Halbersma 2018-2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from setuptools import setup, find_packages

setup(
    name='kleier',
    version='0.1.0-dev0',
    description='Data science tools for the Kleier archive of Stratego results',
    url='https://github.com/rhalbersma/kleir',
    author='Rein Halbersma',
    license='Boost Software License 1.0 (BSL-1.0)',
    packages=find_packages(where='src'),
    package_dir={'':'src'},
    package_data={
        'kleier': ['data/*.pkl'],
    },    
    install_requires=[
        'bs4', 'lxml', 'numpy', 'pandas', 'requests'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
