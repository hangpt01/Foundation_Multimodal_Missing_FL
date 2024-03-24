python main.py \
    --task food101_classification_cnum1_dist0_skew0_seed0_centralized_no_missing \
    --model baseline \
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
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 256 \
    --test_batch_size 256 \
    --gpu 0 \
    --wandb