python generate_fedtask.py \
    --benchmark ptbxl_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 20 \
    --seed 0 \
    --missing \
    --modal_missing_case4

python main.py \
    --task ptbxl_classification_cnum20_dist0_skew0_seed0_missing_modal_case4 \
    --model fedmsplit_gaga_c6_contrastive \
    --algorithm multimodal.ptbxl_classification.fedmsplit_gaga \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --proportion 1.0 \
    --num_epochs 3 \
    --learning_rate 0.5 \
    --lr_scheduler 0 \
    --learning_rate_decay 1 \
    --batch_size 64 \
    --test_batch_size 64 \
    --gpu 1 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --wandb