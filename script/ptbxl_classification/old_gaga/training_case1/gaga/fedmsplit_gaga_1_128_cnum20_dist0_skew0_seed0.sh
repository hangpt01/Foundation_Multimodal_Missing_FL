python generate_fedtask.py \
    --benchmark ptbxl_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 20 \
    --seed 0 \
    --missing

python main.py \
    --task ptbxl_classification_cnum20_dist0_skew0_seed0_missing \
    --model fedmsplit_gaga \
    --algorithm multimodal.ptbxl_classification.fedmsplit \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --proportion 1.0 \
    --num_epochs 1 \
    --learning_rate 0.5 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --gpu 1 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --wandb