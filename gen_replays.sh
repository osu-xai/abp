#!/bin/bash

# name the job
#$ -N abp_gen_replays
#$ -q eecs,eecs2,share,share2,share3,share4

#$ -o gen_replays.out
#$ -j y
for i in `seq 1 30`;
do
   python -m abp.trainer.task_runner -t abp.examples.scaii.city_attack.bad -f ./tasks/sky-rts/city-attack/hra/v1 -v --eval
   if [ $? -eq 0 ] 
   then
       mv ~/.scaii/replays/replay.scr ~/.scaii/replays/candidate_${i}.scr
   fi
done
