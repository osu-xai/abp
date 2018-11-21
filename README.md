Adaptation Based Programming
============================

## Installation
```bash
    git clone https://github.com/osu-xai/abp
    cd abp
    pip3 install -r requirements.txt
    python3 setup.py install
```

## Testing

Run unit and integration tests with:

```bash
    ./test.sh
```

## Usage

### Local Run
* In a separate terminal, start Visdom and keep it running:

```
visdom
```

Visdom will display images during training/evaluation in a web
interface.
By default, Visdom will serve at `http://localhost:8097`.

* Train :
    ```bash
    python -m abp.trainer.task_runner
    ```

    It takes the following arguments:

    | Command Options          | Description        |
    |------------------|--------------------|
    |--example EXAMPLE | The example to run |
    |--adaptive ADAPTIVE|The adaptive to use for the run|
    |--job-dir JOB_DIR  | The location to write tensorflow summaries|
    |--model-path MODEL_PATH |The location to save the model |
    |--restore-model      | Restore the model instead of training a new model|
    |-r, --render       |   Set if it should render the test episodes|
    |--training-episodes TRAINING_EPISODES| Set the number of training episodes|
    |--test-episodes TEST_EPISODES| Set the number of test episodes|
    |--decay-steps DECAY_STEPS | Set the decay rate for exploration|

    Minimal Example:
    ```bash
    python -m abp.trainer.task_runner -f tasks/fruit_collection/hra/v1 -t abp.examples.open_ai.fruit_collection.hra -r
    ```

    Example using StarCraft II:
    ```bash
    python -m abp.trainer.task_runner -f tasks/four_towers/hra/v1 -t abp.examples.sc2env.four_towers.hra
    python -m abp.trainer.task_runner -f tasks/four_towers_friendly_units/hra/v1 -t abp.examples.pysc2.four_towers_friendly_units.hra
    ```


* Visualize Results:
    ```tensorboard --logdir=tensorflow_summaries```


### Using CloudML
To run the job using cloud ML use the following commands

#### Setup Job Parameters
```
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="abp_tictactoe_$now"
TRAINER_PACKAGE_PATH="/path-to-abp/abp"
TRAINER_CONFIG_PATH="/path-to-abp/abp/trainer/cloudml-gpu.yml"
MAIN_TRAINER_MODULE="abp.trainer.task_runner"
JOB_DIR="gs://path-to-job-dir"
MODEL_PATH="gs://path-to-model"
EXAMPLE="tictactoe"
ADAPTIVE="hra"
TRAINING_EPISODES=1000
DECAY_STEPS=250

```

#### Submit the job
```
gcloud ml-engine jobs submit training $JOB_NAME \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --job-dir $JOB_DIR \
    --region us-east1 \
    --stream-logs \
    --config=$TRAINER_CONFIG_PATH \
    -- \
    --example $EXAMPLE \
    --adaptive $ADAPTIVE \
    --model-path $MODEL_PATH \
    --training-episodes $TRAINING_EPISODES \
    --decay-steps $DECAY_STEPS
```
