#!/bin/bash
set -euo pipefail

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Compiler
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# GPU arch — for H100, use sm_90 only
export TORCH_CUDA_ARCH_LIST="9.0"

# ABI compatibility (PyTorch 2.6.0+cu124 uses ABI=1)
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
export PATH=/DATA/anaconda3/envs/nkimi/bin:$PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Recommended memory fragmentation setting
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6,expandable_segments:True

DIR=`pwd`

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6001}

# Use local/pretrained path here (avoid specifying HF model and local path together)
PRETRAINED_MODEL_PATH="/DATA/nfsshare/Adarsh/KIMI/models/Kimi-Audio-7B"
DATA=""

function usage() {
    echo 'Usage: bash finetune_ds.sh -m PRETRAINED_MODEL_PATH -d DATA_PATH'
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model_path) PRETRAINED_MODEL_PATH="$2"; shift ;;
        -d|--data) DATA="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument $1"; usage; exit 1 ;;
    esac
    shift
done

if [ -z "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: PRETRAINED_MODEL_PATH empty"
    usage
    exit 1
fi

if [ -z "$DATA" ]; then
    echo "Error: DATA empty"
    usage
    exit 1
fi

if [ ! -f "$DATA" ]; then
    echo "Error: DATA file does not exist: $DATA"
    exit 1
fi

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: PRETRAINED_MODEL_PATH does not exist: $PRETRAINED_MODEL_PATH"
    exit 1
fi

echo "PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "DATA: $DATA"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

echo "start finetune"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_path "$PRETRAINED_MODEL_PATH" \
    --data_path "$DATA" \
    --eval_ratio 0.05 \
    --bf16 True \
    --output_dir /DATA/nfsshare/Adarsh/KIMI/output \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune_codes/ds_config_zero3.json \
    --resume_from_checkpoint /DATA/nfsshare/Adarsh/KIMI/output/checkpoint-100000

