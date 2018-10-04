For ($i = 0; $i -le 20; $i++) {
    try {
        python -m abp.trainer.task_runner -t abp.examples.scaii.city_attack.bad -f ./tasks/sky-rts/city-attack/hra/bad --eval
        Move-Item -Path C:\Users\zoe\.scaii\replays\replay.scr -Destination C:\Users\zoe\.scaii\replays\candidate_${i}.scr
    }
    catch {
        echo "Failed"
    }
}