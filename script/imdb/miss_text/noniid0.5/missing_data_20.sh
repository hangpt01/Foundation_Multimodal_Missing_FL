python main.py \
    --task imdb_classification_cnum20_dist1_skew0.5_seed0_missing_ratio_0.7_0.7_missing_type_text_text_both_ratio_0.0 \
    --model get_missing_data \
    --algorithm multimodal.imdb_classification.get_missing_data \
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
    --batch_size 64 \
    --test_batch_size 64 \
    --max_text_len 128 \
    --gpu 0 

