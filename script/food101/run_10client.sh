python main.py \
    --task food101_classification_cnum10_dist0_skew0_seed0_missing_each_0.2 \
    --model baseline \
    --algorithm multimodal.food101_classification.baseline \
    --sample full \
    --aggregate other \
    --num_rounds 2 \
    --early_stop 2  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.01 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 32 \
    --test_batch_size 32 \
    --gpu 0 
    # \
    # --wandb