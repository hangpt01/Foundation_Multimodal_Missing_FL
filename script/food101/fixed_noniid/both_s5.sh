python main.py \
    --task food101_8_classes_fixed_cnum20_dist1_skew0.1_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
    --model proposal \
    --algorithm multimodal.food101_8_classes_fixed.proposal \
    --sample full \
    --aggregate other \
    --num_rounds 250 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 5678 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.05 \
    --reduce_sim_scalar 0.005 \
    --num_epochs 1 \
    --num_outer_loops 5 \
    --learning_rate_decay 1.0 \
    --note rebuttal \
    --batch_size 128 \
    --test_batch_size 128 \
    --max_text_len 40 \
    --gpu 0 \
    --wandb