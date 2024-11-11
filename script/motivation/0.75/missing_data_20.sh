python main.py \
    --task food101_motivation_cnum20_dist0_skew0_seed0_missing_ratio_0.75_0.75_missing_type_both_both_both_ratio_0.5 \
    --model get_missing_data \
    --algorithm multimodal.food101_motivation.get_missing_data \
    --sample full \
    --aggregate other \
    --num_rounds 250 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.01 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --note loadvilt \
    --batch_size 256 \
    --test_batch_size 256 \
    --max_text_len 40 \
    --gpu 1 
    # \
    # --wandb
