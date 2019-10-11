#          Copyright Rein Halbersma 2019.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import pandas as pd

import kleier.utils

import player, event

def finalize_player_index(player_index: pd.DataFrame, event_table: pd.DataFrame, player_cross: pd.DataFrame) -> pd.DataFrame:
    pid_name = player_index
    sur_pre_nat_name = (event_table
        .loc[:, ['sur', 'pre', 'nat']]
        .drop_duplicates()
        .assign(name = lambda x: x.pre + ' ' + x.sur))
    pid_sur_pre_nat = pd.merge(
        pid_name, sur_pre_nat_name,
        how='outer', indicator=True, validate='one_to_one')
    pid_sur_pre_no_nat = (pid_sur_pre_nat
        .query('_merge != "both"')
        .assign(
            sur = lambda x: x.name.str.split(expand=True)[1],
            pre = lambda x: x.name.str.split(expand=True)[0])
        .drop(columns=['name', '_merge']))
    pid_sur_pre_nat = (pid_sur_pre_nat
        .query('_merge == "both"')
        .drop(columns=['name', '_merge'])
        .append(pid_sur_pre_no_nat)
        .sort_values('pid'))
    name = (player_cross
        .loc[:, ['name1']]
        .rename(columns=lambda x: x[:-1])
        .drop_duplicates())
    sur_pre_rating_name = (player_cross
        .loc[:, ['sur2', 'pre2', 'rating2']]
        .rename(columns=lambda x: x[:-1])
        .drop_duplicates()
        .assign(name = lambda x: x.pre + ' ' + x.sur))
    sur_pre_rating = (pd.merge(
        name, sur_pre_rating_name,
        how='outer', validate='one_to_one')
        .drop(columns=['name']))
    pid_sur_pre_nat_rating = pd.merge(
        pid_sur_pre_nat, sur_pre_rating,
        how='outer', validate='one_to_one')
    return pid_sur_pre_nat_rating

def event_table_add_pid(event_table: pd.DataFrame, player_index: pd.DataFrame) -> pd.DataFrame:
    eid_gid_rank_sur_pre_nat = event_table
    pid_sur_pre_nat = (player_index
        .drop(columns=['rating']))
    eid_gid_rank_pid = (pd
        .merge(eid_gid_rank_sur_pre_nat, pid_sur_pre_nat)
        .sort_values(['eid', 'gid', 'rank'])
        .reset_index(drop=True))
    columns = eid_gid_rank_pid.columns.to_list()
    columns = columns[:3] + [columns[-1]] + columns[3:-1]
    return eid_gid_rank_pid[columns]

def event_cross_add_player(event_cross: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    rank1_rank2 = event_cross
    eid_cols = ['eid', 'gid']
    pid_cols = ['rank', 'pid', 'sur', 'pre', 'nat', 'rating']
    rank_pid = (event_table
        .loc[:, eid_cols + pid_cols])
    rank1_pid1 = rank_pid.rename(columns={c: c + '1' for c in pid_cols})
    rank2_pid2 = rank_pid.rename(columns={c: c + '2' for c in pid_cols})
    rank1_rank2_pid1 = (pd
        .merge(rank1_rank2, rank1_pid1, how='left'))
    rank1_rank2_pid1_pid2 = (pd.
        merge(rank1_rank2_pid1, rank2_pid2, how='left'))

def main():
    player.download()
    event.download()

    player_index = kleier.load_dataset('player_index')
    player_cross = kleier.load_dataset('player_cross')
    event_index  = kleier.load_dataset('event_index')
    event_table  = kleier.load_dataset('event_table')
    event_cross  = kleier.load_dataset('event_cross')

    player_index = finalize_player_index(player_index, event_table, player_cross)
    event_table = event_table_add_pid(event_table, player_index)
