python main.py \
    --task food101_classification_arrow_cnum20_dist0_skew0_seed0_missing_each_0.25 \
    --model missing_aware \
    --algorithm multimodal.food101_classification_arrow.missing_aware \
    --sample full \
    --aggregate other \
    --num_rounds 3 \
    --early_stop 3  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.01 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 88 \
    --test_batch_size 88 \
    --gpu 3 
    # \
    # --wandb