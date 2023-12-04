python generate_fedtask.py \
    --benchmark ptbxl_classification_lm \
    --dist 0 \
    --skew 0.0 \
    --num_clients 20 \
    --seed 0 \
    --missing \

python main.py \
    --task ptbxl_classification_lm_cnum20_dist0_skew0_seed0_missing_mifl_local_missing \
    --model mifl_ps08_pm03 \
    --algorithm multimodal.ptbxl_classification_lm.mifl \
    --sample full \
    --aggregate other \
    --num_rounds 300 \
    --proportion 1.0 \
    --num_epochs 3 \
    --learning_rate 0.5 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --batch_size 256 \
    --test_batch_size 256 \
    --gpu 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --ps 0.8 \
    --pm 0.3 \
    --wandb