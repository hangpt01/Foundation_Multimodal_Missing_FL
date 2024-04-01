python main.py \
    --task food101_classification_arrow_cnum1_dist0_skew0_seed0_centralized_no_missing \
    --model centralized \
    --algorithm multimodal.food101_classification_arrow.fedavg_no_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --early_stop 20  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.005 \
    --num_epochs 1 \
    --learning_rate_decay 0.9 \
    --batch_size 88 \
    --test_batch_size 88 \
    --gpu 0 \
    --wandb