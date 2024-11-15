python generate_fedtask.py \
    --benchmark imdb_motivation \
    --dist 0 \
    --skew 0 \
    --num_clients 1 \
    --seed 0 \
    --missing \
    --missing_ratio_train 0 \
    --missing_ratio_test 0 \
    --missing_type_train both \
    --missing_type_test both \
    --both_ratio 0.5 \
    --max_text_len 128
python main.py \
    --task imdb_motivation_cnum20_dist0_skew0_seed0_missing_ratio_0.0_0.0_missing_type_both_both_both_ratio_0.5 \
    --model L2P_Prob_prompt_only_global_no_print \
    --algorithm multimodal.imdb_motivation.L2P_Prob_prompt_only_global \
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
    --note fixbug \
    --batch_size 256 \
    --test_batch_size 256 \
    --max_text_len 128 \
    --gpu 0 
    # \
    # --wandb