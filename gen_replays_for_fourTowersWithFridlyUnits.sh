#!/bin/bash

# name the job
#$ -N abp_gen_replays
#$ -q eecs,eecs2,share,share2,share3,share4

#$ -o gen_replays.out
#$ -j y
for i in `seq 1 100`;
do
    python3 -m abp.trainer.task_runner -f tasks/four_towers_friendly_units/hra/v1 -t abp.examples.pysc2.four_towers_friendly_units.hra
done
