python generate_fedtask.py \
    --benchmark mhd_reduce_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 50 \
    --seed 0 \
    --missing

python main.py \
    --task mhd_reduce_classification_cnum50_dist0_skew0_seed0_missing \
    --model add_element_wise \
    --algorithm multimodal.mhd_reduce_classification.fedavg_add_element_wise \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --proportion 1.0 \
    --num_epochs 2 \
    --optimizer Adam \
    --learning_rate 0.005 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --batch_size 64 \
    --test_batch_size 64 \
    --gpu 1 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.0 \
    --wandb