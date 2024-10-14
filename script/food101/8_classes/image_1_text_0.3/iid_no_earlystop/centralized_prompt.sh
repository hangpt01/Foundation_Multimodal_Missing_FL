python main.py \
    --task food101_classification_8_classes_cnum1_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_text_text_both_ratio_0.0 \
    --model centralized_prompt \
    --algorithm multimodal.food101_classification_8_classes.centralized_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 250 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 512 \
    --test_batch_size 512 \
    --max_text_len 256 \
    --gpu 0 \
    --wandb