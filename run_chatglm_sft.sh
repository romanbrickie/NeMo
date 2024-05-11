pkill -9 python

WORK=$(cd $(dirname $0); pwd)
MODEL='/lustre/fsw/general_sa/xueh/nemo24.03_chatglm/chatglm3-6b.nemo'
TRAIN_DS="[/lustre/fsw/general_sa/xueh/nemo24.03_chatglm/AdvertiseGen/train.json]"
VALID_DS="[/lustre/fsw/general_sa/xueh/nemo24.03_chatglm/AdvertiseGen/dev.json]"
CONCAT_SAMPLING_PROBS="[1]"

#WORK=/lustre/fsw/general_sa/xueh/nemo24.03_chatglm/NeMo

CONFIG_DIR=${WORK}/conf
CONFIG_FILE=chatglm_sft.yaml
EXP_DIR=${WORK}/exp
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
TE=True
FP8=False
GLOBAL_BATCH_SIZE=256
MICRO_BATCH_SIZE=1
MAX_SEQ_LENGTH=2048
MAX_STEPS=2000
DO_CHECKPOINT=True
CHECKPOINT_STEPS=1000
SAVE_LAST=False
LR=1e-5
OPT=distributed_fused_adam

# TODO
WBPROJECT=chatglm3-sft-20240411
NAME=chatglm3-6b-bf16

# Necessary Exports
export HYDRA_FULL_ERROR=1
export NCCL_AVOID_RECORD_STREAMS=1

# && pip uninstall megatron_core \
# && pip uninstall nemo \
CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 \
   /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
        --config-path=${CONFIG_DIR} --config-name=${CONFIG_FILE} \
        name=${NAME} \
        trainer.devices=8 \
        trainer.num_nodes=1 \
        trainer.max_steps=${MAX_STEPS} \
        +trainer.detect_anomaly=False \
	exp_manager.exp_dir=${EXP_DIR} \
        exp_manager.name=${NAME} \
        exp_manager.create_checkpoint_callback=${DO_CHECKPOINT} \
        exp_manager.checkpoint_callback_params.save_last=${SAVE_LAST} \
        exp_manager.checkpoint_callback_params.every_n_train_steps=${CHECKPOINT_STEPS} \
        exp_manager.create_wandb_logger=True \
        exp_manager.wandb_logger_kwargs.project=${WBPROJECT} \
        exp_manager.wandb_logger_kwargs.name=${NAME} \
        model.transformer_engine=${TE} \
        model.fp8=${FP8} \
        model.restore_from_path=${MODEL} \
        model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
        model.pipeline_model_parallel_size=${PIPELINE_MODEL_PARALLEL_SIZE} \
        model.optim.name=${OPT} \
        model.optim.lr=${LR} \
        model.micro_batch_size=${MICRO_BATCH_SIZE} \
        model.global_batch_size=${GLOBAL_BATCH_SIZE} \
        model.data.train_ds.file_names=${TRAIN_DS} \
        model.data.validation_ds.file_names=${VALID_DS} \
        model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
        model.data.train_ds.max_seq_length=${MAX_SEQ_LENGTH} \
        model.data.train_ds.context_key='content' \
        model.data.train_ds.truncation_field='summary' \
        model.data.train_ds.label_key='summary' \
	model.data.train_ds.shuffle=False \
        model.data.train_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
        model.data.train_ds.global_batch_size=${GLOBAL_BATCH_SIZE} 2>&1|tee chatglm_sft_bf16.log

