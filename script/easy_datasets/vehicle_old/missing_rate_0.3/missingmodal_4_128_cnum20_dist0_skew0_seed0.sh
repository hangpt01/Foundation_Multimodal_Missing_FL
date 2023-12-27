python generate_fedtask.py \
    --benchmark vehicle_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 23 \
    --seed 0 \
    --missing \
    --modal_missing_case3

python main.py \
    --task vehicle_classification_cnum23_dist0_skew0_seed0_missing_rate_0.3 \
    --model missing_modal \
    --algorithm multimodal.vehicle_classification.missing_modal \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --proportion 1.0 \
    --num_epochs 4 \
    --learning_rate 0.05 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --gpu 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.0 \
    --wandb