python main.py \
    --task food101_classification_8_classes_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
    --model fedavg_no_prompt \
    --algorithm multimodal.food101_classification_8_classes.fedavg_no_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 2 \
    --early_stop 2  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.1 \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 16 \
    --test_batch_size 16 \
    --gpu 0 
    # \
    # --wandb