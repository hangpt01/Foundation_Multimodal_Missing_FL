python main.py \
    --task food101_motivation_cnum20_dist0_skew0_seed0_missing_ratio_0.75_0.75_missing_type_both_both_both_ratio_0.5 \
    --model proposal_pool_30 \
    --algorithm multimodal.food101_motivation.L2P_Prob_prompt_only_global \
    --sample full \
    --aggregate other \
    --num_rounds 150 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --num_outer_loops 5 \
    --learning_rate_decay 1.0 \
    --note ablation_pool \
    --batch_size 128 \
    --test_batch_size 128 \
    --max_text_len 40 \
    --gpu 0 \
    --wandb