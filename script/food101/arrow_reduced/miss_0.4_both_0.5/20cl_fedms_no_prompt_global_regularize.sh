python main.py \
    --task food101_classification_arrow_reduced_cnum20_dist0_skew0_seed0_missing_ratio_0.4_0.4_missing_type_both_both_both_ratio_0.5 \
    --model fedmsplit_no_prompt_global_regularize \
    --algorithm multimodal.food101_classification_arrow_reduced.fedmsplit_no_prompt_global_regularize \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 30  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 64 \
    --test_batch_size 64 \
    --gpu 0 \
    --wandb