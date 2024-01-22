#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

TRAIN_PATH=$1
TEST_PATH=$2
VALID_PATH=$3

PROTTRANS_PATH='./pretrained/prot_t5_xl_uniref50'
ORGANISM_DICTIONARY='./data/new_modeling/demo/organism_token.json'
MOLEBERT_DIR="/fs1/home/wangll1/WJJ/Mole_BERT/model_gin/Mole-BERT.pth"

CHECKPOINT_DIR=$OUT_DIR

EPOCHS=100
LOG_INTERVAL=2
BATCH_SIZE=1

MAX_LENGTH=1024

EXPERT_LAERYS=1
EXPERT_OUT=768
EXPERT_HIDDEN=768
PLE_LAYERS=1

NUM_EXPERTS=4
NUM_TASKS=2
#NUM_TASKS=1
NUM_WORKERS=70

LABEL_INDEX="14 15"
WEIGHTS="1 1"

SEQUENCE_COLUMN=$4
SMILES_COLUMN=$5

ORGANISM_FLAG=$6
ORGANISM_COLUMN=$7

PH_FLAG=$8
PH_COLUMN=$9

TEMPERATURE_FLAG=${10}
TEMPERATURE_COLUMN=${11}

TOWER_HID_LAYER=2
TOWER_HID_UNIT=128
TOWER_DROPOUT="0 0" # kotori

BETA2=0.999

WEIGHT_DECAY=1e-6
LIGAND_ENC_LR=1e-4
PROTEIN_ENC_LR=1e-4
HEAD_LR=1e-4

DEVICE={12}
CHECKPOINT_DIR={13}
OUT_DIR={14}

DROPOUT=0.3

yhrun -N 1 -p gpu1 --gpus-per-node=1 --cpus-per-gpu=4 \
 python train.py \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --valid_path $VALID_PATH \
    --out_dir $OUT_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --sequence_index $SEQUENCE_INDEX \
    --smiles_index $SMILES_INDEX \
    --label_index $LABEL_INDEX \
    --weights $WEIGHTS \
    --max_length $MAX_LENGTH \
    --molclr_path $MOLCLR_PATH \
    --molebert_dir $MOLEBERT_DIR \
    --prottrans_path $PROTTRANS_PATH \
    --esm_dir $ESM_DIR \
    --organism_dictionary $ORGANISM_DICTIONARY \
    --organism_index $ORGANISM_INDEX \
    --ph_index $PH_INDEX \
    --temperature_index $TEMPERATURE_INDEX \
    --atten_heads $ATTEN_HEADS \
    --tower_hid_layer $TOWER_HID_LAYER \
    --tower_hid_unit $TOWER_HID_UNIT \
    --expert_layers $EXPERT_LAERYS \
    --expert_out $EXPERT_OUT \
    --expert_hidden $EXPERT_HIDDEN \
    --ple_layers $PLE_LAYERS \
    --num_experts $NUM_EXPERTS \
    --num_tasks $NUM_TASKS \
    --dropout $DROPOUT \
    --tower_dropout $TOWER_DROPOUT \
    --ligand_enc_lr $LIGAND_ENC_LR \
    --protein_enc_lr $PROTEIN_ENC_LR \
    --head_lr $HEAD_LR \
    --weight-decay $WEIGHT_DECAY \
    --beta2 $BETA2 \
    --epochs $EPOCHS \
    --log_interval $LOG_INTERVAL \
    --frozen_ligand_enc \
    --frozen_prot_enc \
    --use_ple \
    --use_temperature \
    --use_ph \
    --use_molebert
    # --use_organism \
    #--use_esm2
    # --use_add
    # --lr_warmup
