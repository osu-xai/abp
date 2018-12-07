#!/bin/bash

# name the job
#$ -N abp_gen_replays
#$ -q eecs,eecs2,share,share2,share3,share4

#$ -o gen_replays.out
#$ -j y
for i in `seq 1 100`;
do
    python3 -m sc2env.play_four_towers_friendly_unit -f ../abp/tasks/four_towers_friendly_units/hra/v1/ -m FourTowersWithFriendlyUnits
done
