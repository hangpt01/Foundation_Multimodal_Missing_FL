python main.py \
    --task food101_classification_arrow_reduced_cnum20_dist0_skew0_seed0_missing_each_0.25 \
    --model fedavg_no_prompt \
    --algorithm multimodal.food101_classification_arrow_reduced.fedavg_no_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 300 \
    --early_stop 20  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.5 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 64 \
    --test_batch_size 64 \
    --gpu 0 
    # \
    # --wandb