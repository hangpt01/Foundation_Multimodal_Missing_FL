python main.py \
    --task ptbxl_reduce_ablation_cnum20_dist0_skew0_seed0_missing_all_6 \
    --model mifl \
    --algorithm multimodal.ptbxl_reduce_ablation.mifl \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --contrastive_weight 0 \
    --learning_rate 0.5 \
    --num_epochs 3 \
    --learning_rate_decay 1.0 \
    --batch_size 64 \
    --test_batch_size 64 \
    --gpu 0 \
    --wandb