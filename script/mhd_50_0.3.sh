python generate_fedtask.py \
    --benchmark mhd_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 50 \
    --seed 0

python main.py \
    --task mhd_classification_cnum50_dist0_skew0_seed0 \
    --model mm \
    --algorithm mm_mhd_fedavg \
    --sample uniform \
    --aggregate weighted_com \
    --num_rounds 100 \
    --proportion 0.3 \
    --num_epochs 5 \
    --learning_rate 0.01 \
    --batch_size 64 \
    --gpu 0 \
    --seed 1234 \
    --test_batch_size 64