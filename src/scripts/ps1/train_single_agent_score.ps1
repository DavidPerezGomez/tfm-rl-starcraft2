$EXPERIMENT_NAME = "2024-09-12"

$SRC_DIR = (get-item $PSScriptRoot).parent.parent.FullName
$BASE_MODELS_DIR = Join-Path (get-item $SRC_DIR).parent.FullName "models"

$AGENT_TYPE = "single"
$AGENT_ALGORITHM = "dqn"
$REWARD_METHOD = "score"

$BASE_MODEL_ID = "single_dqn"
$MAP = "Simple64"

$TRAIN_EPISODES = 100
$EPSILON_DECAY = 0.98 # 300 EP
$LOG_SUFFIX = "_01"
$LEARNING_RATE_MILESTONES = "40 70 90"
$LEARNING_RATE = 0.001
$DQN_SIZE = "large" # extra_small, small, medium, large, extra_large
$MEMORY_SIZE = 100000
$BURN_IN = 10000
$BATCH_SIZE = 512

$AGENT_KEY = -join($AGENT_TYPE, ".", $AGENT_ALGORITHM)
$MODELS_DIR = Join-Path $BASE_MODELS_DIR $EXPERIMENT_NAME

$MODEL_ID = -join($BASE_MODEL_ID, "_", $REWARD_METHOD)

$MODEL_DIR = Join-Path $MODELS_DIR $MODEL_ID

New-Item -ItemType Directory -Force -Path "$MODEL_DIR"

Write-Host "Training ${MAP}"
New-Item -ItemType File -Force -Path $(Join-Path "$MODEL_DIR" $(-join("_01_training_start_", $TRAIN_EPISODES, "_ep")))

python "$(Join-Path $SRC_DIR "runner.py")" `
    --agent_key "$AGENT_KEY" `
    --map_name "$MAP" `
    --num_episodes $TRAIN_EPISODES `
    --log_file "$(Join-Path $MODEL_DIR $(-join("training", $LOG_SUFFIX, ".log")))" `
    --model_id $MODEL_ID `
    --models_path "$MODELS_DIR" `
    --epsilon_decay $EPSILON_DECAY `
    --lr_milestones "$LEARNING_RATE_MILESTONES" `
    --lr $LEARNING_RATE `
    --dqn_size $DQN_SIZE `
    --memory_size $MEMORY_SIZE `
    --burn_in $BURN_IN `
    --batch_size $BATCH_SIZE `
    --reward_method $REWARD_METHOD `
| Set-Content -Path $(Join-Path $MODEL_DIR $(-join($MAP, $LOG_SUFFIX, ".log")))

New-Item -ItemType File -Force -Path $(Join-Path "$MODEL_DIR" $(-join("_02_training_done_", $TRAIN_EPISODES, "_ep")))
