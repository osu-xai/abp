For ($i = 0; $i -le 20; $i++) {
    try {
        python -m abp.trainer.task_runner -t abp.examples.scaii.city_attack.bad -f ./tasks/sky-rts/city-attack/hra/bad
    }
    catch {
        echo "Repeating"
    }
}