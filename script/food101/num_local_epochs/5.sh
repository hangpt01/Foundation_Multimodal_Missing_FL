python main.py \
    --task food101_motivation_cnum20_dist0_skew0_seed0_missing_ratio_0.5_0.5_missing_type_both_both_both_ratio_0.5 \
    --model L2P_Prob_prompt_only_global_no_print \
    --algorithm multimodal.food101_motivation.L2P_Prob_prompt_only_global \
    --sample full \
    --aggregate other \
    --num_rounds 250 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.05 \
    --num_epochs 5 \
    --num_outer_loops 5 \
    --learning_rate_decay 1.0 \
    --note change_percentage_client \
    --batch_size 512 \
    --test_batch_size 512 \
    --max_text_len 40 \
    --gpu 0 \
    --wandb