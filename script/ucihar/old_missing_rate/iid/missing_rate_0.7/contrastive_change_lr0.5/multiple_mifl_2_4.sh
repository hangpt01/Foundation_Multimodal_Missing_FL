for contrastive_weight in 2 4; do
    taskset -c 1-40 python main_w_contrastive.py \
        --task ucihar_classification_cnum20_dist0_skew0_seed0_missing_rate_0.7 \
        --model mifl \
        --algorithm multimodal.ucihar_classification.mifl \
        --sample full \
        --aggregate other \
        --num_rounds 1000 \
        --early_stop 50  \
        --proportion 1.0 \
        --lr_scheduler 0 \
        --seed 1234 \
        --fedmsplit_prox_lambda 0.01 \
        --contrastive_weight $contrastive_weight \
        --learning_rate 0.5 \
        --num_epochs 3 \
        --learning_rate_decay 1.0 \
        --batch_size 128 \
        --test_batch_size 128 \
        --gpu 3 \
        --wandb
done