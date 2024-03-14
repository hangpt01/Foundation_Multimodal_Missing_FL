python main.py \
    --task food101_classification_cnum1_dist0_skew0_seed0_missing \
    --model baseline \
    --algorithm multimodal.food101_classification.baseline \
    --sample full \
    --aggregate other \
    --num_rounds 1 \
    --early_stop 2  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.1 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 40 \
    --gpu 0 
    # \
    # --wandb