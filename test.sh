#!/bin/bash
set -euxo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m ragen.eval \
    --config-name eval \
    model_path=./checkpoints/dual_qwen_4b_thinking \
    trainer.n_gpus_per_node=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    es_manager.train.env_configs.tags='["MATH"]' \
    es_manager.train.env_configs.n_groups='[8]' \
    es_manager.val.env_configs.tags='["MATH"]' \
    es_manager.val.env_configs.n_groups='[32]'