#!/bin/bash
mkdir -p ~/.scaii/git/SCAII/replays/city-attack-v1/
for i in `seq 1 30`;
do
   python -m abp.trainer.task_runner -t abp.examples.scaii.city_attack.hra -f ./tasks/sky-rts/city-attack/hra/v1 -v --eval
   if [ $? -eq 0 ] 
   then
       cp ~/.scaii/replays/replay.scr ~/.scaii/git/SCAII/replays/city-attack-v1/candidate_${i}.scr
   fi
done
