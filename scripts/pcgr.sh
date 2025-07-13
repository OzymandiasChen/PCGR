

GPU_ID=0
DATE=1005
EXP=100500
ORDER=order_0
SEED=1


# PCGR

source ./enviornment
cd $SnD_DIR
mkdir -p $MODELS_DIR/$DATE/$EXP
mkdir -p $LOGS_DIR/$DATE/$EXP



CUDA_VISIBLE_DEVICES=$GPU_ID \
python train.py \
    --expname $EXP --seed $SEED \
    --max_sample_len_flag \
    --check_during_pseudo_gene_flag  \
    --vae_type prototype_cvae --vae_train_input_type 0 \
    --share_params --mlp_bottleneck_dim 128 \
    --add_kd --KD_term 1.0 --KD_temperature 2.0 \
    --kl_loss_weight 0.5 --lm_loss_weight 0.5 \
    --sta_flag  \
    --semantic_drift_flag  --semantic_drift_weight 1 --sigma_drift 4 \
    --logvar_drift_flag  --logvar_semantic_drift_weight 1 --logvar_sigma_drift 4 \
    --self_arguement_flag  --self_arguement_weight 1.0 \
    --order $ORDER \
    --pseudo_data_ratio 0.2 --alpha_z 1.0 --z_dim 128 \
    --memory_size 50 \
    --top_k 100 --top_p 0.9 \
    --ctx_max_len 192 --ctx_max_ans_len 128 \
    --num_cycle 4 \
    --fp16 --gpt2_path $GPT2_LARGE_DIR \
    --eval_steps 80 \
    --train_batch_size 8 --gene_batch_size 8 \
    --data_dir $PROJECT_DIR/datasets \
    --output_dir $MODELS_DIR/$DATE/$EXP --memory_path $MODELS_DIR/$DATE/$EXP \
    --tb_log_dir $MODELS_DIR/$DATE/$EXP/tensorboard \
    > $LOGS_DIR/$DATE/$EXP/train.log 2>&1


python final_score.py \
  --expname $EXP --order $ORDER \
  --res_dir $MODELS_DIR/$DATE/$EXP --output_dir $MODELS_DIR/$DATE/$EXP \
  > $LOGS_DIR/$DATE/$EXP/score.log 2>&1



    