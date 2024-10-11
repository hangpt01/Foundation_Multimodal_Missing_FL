python main.py \
    --task imdb_classification_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_text_text_both_ratio_0.0 \
    --model Agglo_mean \
    --algorithm multimodal.imdb_classification.Agglo_mean \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.01 \
    --num_epochs 1 \
    --num_outer_loops 5 \
    --learning_rate_decay 1.0 \
    --note 3clusters_loadvilt \
    --batch_size 256 \
    --test_batch_size 256 \
    --max_text_len 128 \
    --gpu 0 \
    --wandb