python main.py \
    --task imdb_classification_cnum1_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_text_text_both_ratio_0.0 \
    --model missing_aware \
    --algorithm multimodal.imdb_classification.missing_aware \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.01 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --max_text_len 128 \
    --gpu 0 \
    --wandb
