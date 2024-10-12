python main.py \
    --task food101_classification_cnum20_dist0_skew0_seed0_missing_each_0.25_10_classes \
    --model baseline_no_miss \
    --algorithm multimodal.food101_classification.baseline \
    --sample full \
    --aggregate other \
    --num_rounds 300 \
    --early_stop 30  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.01 \
    --num_epochs 3 \
    --learning_rate_decay 1.0 \
    --batch_size 40 \
    --test_batch_size 40 \
    --gpu 0 \
    --wandb