python generate_fedtask.py \
    --benchmark ptbxl_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 20 \
    --seed 0 \
    --missing

python main.py \
    --task ptbxl_classification_cnum20_dist0_skew0_seed0_missing \
    --model fedmsplit_gaga_c1_transformer \
    --algorithm multimodal.ptbxl_classification.fedmsplit_gaga_transformer \
    --sample full \
    --aggregate other \
    --num_rounds 3 \
    --proportion 1.0 \
    --num_epochs 1 \
    --learning_rate 0.5 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --batch_size 32 \
    --test_batch_size 32 \
    --gpu 1 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 
    # \
    # --wandb