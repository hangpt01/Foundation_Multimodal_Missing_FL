python main.py \
    --task food101_classification_8_classes_cnum20_dist1_skew0.5_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
    --model fedmsplit_prompt_no_regularize \
    --algorithm multimodal.food101_classification_8_classes.fedmsplit_prompt_no_regularize \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 64 \
    --test_batch_size 64 \
    --gpu 1 \
    --wandb