#          Copyright Rein Halbersma 2019-2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from setuptools import setup, find_packages

setup(
    name='kleier',
    version='0.1.0-dev0',
    description='Data science tools for the Kleier archive of Stratego results',
    url='https://github.com/rhalbersma/kleier',
    author='Rein Halbersma',
    license='Boost Software License 1.0 (BSL-1.0)',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points = {
        'console_scripts': [
            'kleier=scripts.cli:kleier',
            'extract=scripts.cli:extract'
            'transform=scripts.cli:transform'
        ],
    },
    install_requires=[
        'bs4', 'click', 'jax', 'jaxlib', 'lxml', 'numpy', 'pandas', 'requests', 'scipy'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha'
        'Intended Audience :: Science/Research'
        'License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)'
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
