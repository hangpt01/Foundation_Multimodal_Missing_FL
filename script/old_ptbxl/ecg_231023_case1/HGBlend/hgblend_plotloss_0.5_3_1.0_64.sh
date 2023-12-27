python generate_fedtask.py \
    --benchmark ptbxl_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 20 \
    --seed 0 \
    --missing

python main.py \
    --task ptbxl_classification_cnum20_dist0_skew0_seed0_missing_mifl_gblend \
    --model hierarchical_gblend \
    --algorithm multimodal.ptbxl_classification.hierarchical_gblend \
    --sample full \
    --aggregate other \
    --num_rounds 20 \
    --early_stop 4  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.5 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 256 \
    --test_batch_size 256 \
    --gpu 1 \
    --wandb