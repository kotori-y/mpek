#!/bin/bash

TRAIN_NORMALIZER_PATH='./checkpoints/normlizer.pickle'
ORGANISM_DICTIONARY='./pretrained/organism_token.json'

PROTTRANS_PATH='./checkpoints/prot_t5_xl_uniref50'
MOLEBERT_DIR="./checkpoints/Mole_BERT/model_gin/Mole-BERT.pth"

BATCH_SIZE=1
# MMOE_UNITS=512
MAX_LENGTH=1024
EXPERT_LAERYS=1
EXPERT_OUT=768
EXPERT_HIDDEN=768
PLE_LAYERS=1

NUM_EXPERTS=5
NUM_TASKS=2
#NUM_TASKS=1
NUM_WORKERS=10

SEQUENCE_COLUMN=$3
SMILES_COLUMN=$4

ORGANISM_FLAG=1
ORGANISM_COLUMN=$5

PH_FLAG=1
PH_COLUMN=$6

TEMPERATURE_FLAG=1
TEMPERATURE_COLUMN=$7

TOWER_HID_LAYER=2
TOWER_HID_UNIT=128
TOWER_DROPOUT="0 0"

DEVICE=$2
# DEVICE='cpu'

DROPOUT=0.2

if [ $ORGANISM_FLAG -eq 1 ] && [ $PH_FLAG -eq 1 ] && [ $TEMPERATURE_FLAG -eq 1 ]; then
  CHECKPOINT_DIR='./checkpoints/mpek_all_conditions.pt'
elif [ $ORGANISM_FLAG -eq 1 ] && [ $PH_FLAG -eq 0 ] && [ $TEMPERATURE_FLAG -eq 0 ]; then
  CHECKPOINT_DIR='./checkpoints/mpek_organism_condition.pt"'
elif [ $ORGANISM_FLAG -eq 0 ] && [ $PH_FLAG -eq 1 ] && [ $TEMPERATURE_FLAG -eq 0 ]; then
  CHECKPOINT_DIR='./checkpoints/mpek_pH_condition.pt'
elif [ $ORGANISM_FLAG -eq 0 ] && [ $PH_FLAG -eq 0 ] && [ $TEMPERATURE_FLAG -eq 1 ]; then
  CHECKPOINT_DIR='./checkpoints/mpek_temperature_condition.pt'
elif [ $ORGANISM_FLAG -eq 1 ] && [ $PH_FLAG -eq 1 ] && [ $TEMPERATURE_FLAG -eq 0 ]; then
  CHECKPOINT_DIR='./checkpoints/mpek_organism_pH_conditions.pt'
elif [ $ORGANISM_FLAG -eq 1 ] && [ $PH_FLAG -eq 0 ] && [ $TEMPERATURE_FLAG -eq 1 ]; then
  CHECKPOINT_DIR='./checkpoints/mpek_organism_temperature_conditions.pt'
elif [ $ORGANISM_FLAG -eq 0 ] && [ $PH_FLAG -eq 1 ] && [ $TEMPERATURE_FLAG -eq 1 ]; then
  CHECKPOINT_DIR='./checkpoints/mpek_pH_temperature_conditions.pt'
else
  CHECKPOINT_DIR='./checkpoints/mpek_all_conditions.pt'
fi


###
# RESULT_FILE=$ROOT'/chenchang_results.csv'

TEST_DATA=$1
###

# nvidia-smi -l 5 2>&1 >gpu_info.log & #检测主节点gpu使用情况

  python evaluate.py \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --train_normalizer_path $TRAIN_NORMALIZER_PATH \
    --test_path $TEST_DATA \
    --sequence_column $SEQUENCE_COLUMN \
    --smiles_column $SMILES_COLUMN \
    --max_length $MAX_LENGTH \
    --molebert_dir $MOLEBERT_DIR \
    --prottrans_path $PROTTRANS_PATH \
    --organism_dictionary $ORGANISM_DICTIONARY \
    --organism_column $ORGANISM_COLUMN \
    --ph_column $PH_COLUMN \
    --temperature_column $TEMPERATURE_COLUMN \
    --tower_hid_layer $TOWER_HID_LAYER \
    --tower_hid_unit $TOWER_HID_UNIT \
    --expert_layers $EXPERT_LAERYS \
    --expert_out $EXPERT_OUT \
    --expert_hidden $EXPERT_HIDDEN \
    --num_experts $NUM_EXPERTS \
    --num_tasks $NUM_TASKS \
    --dropout $DROPOUT \
    --tower_dropout $TOWER_DROPOUT \
    --use_organism $ORGANISM_FLAG \
    --use_ph $PH_FLAG \
    --use_temperature $TEMPERATURE_FLAG \
    --checkpoint_dir $CHECKPOINT_DIR
