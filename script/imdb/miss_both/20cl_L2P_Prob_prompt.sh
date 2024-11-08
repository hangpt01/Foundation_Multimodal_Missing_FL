python main.py \
    --task imdb_classification_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
    --model L2P_Prob_prompt \
    --algorithm multimodal.imdb_classification.L2P_Prob_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 250 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.01 \
    --num_epochs 1 \
    --num_outer_loops 5 \
    --learning_rate_decay 1.0 \
    --note loadvilt \
    --batch_size 256 \
    --test_batch_size 256 \
    --max_text_len 128 \
    --gpu 0 
    # \
    # --wandb