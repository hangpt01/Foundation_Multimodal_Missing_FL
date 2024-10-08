python main.py \
    --task food101_classification_8_classes_cnum1_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
    --model centralized_no_prompt \
    --algorithm multimodal.food101_classification_8_classes.fedavg_no_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.01 \
    --num_epochs 1 \
    --learning_rate_decay 1 \
    --batch_size 512 \
    --test_batch_size 512 \
    --gpu 1 
    # \
    # --wandb