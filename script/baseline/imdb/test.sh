python main.py \
    --task imdb_baselines_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
    --model fedadam \
    --algorithm multimodal.imdb_baselines.fedadam \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.1 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --note loadvilt \
    --batch_size 512 \
    --test_batch_size 512 \
    --max_text_len 128 \
    --gpu 0 
    # \
    # --wandb
