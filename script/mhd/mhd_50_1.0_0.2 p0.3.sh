python generate_fedtask.py \
    --benchmark mhd_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 50 \
    --seed 0 \
    --percentages 0.2 0.2

python main.py \
    --task mhd_classification_cnum50_dist0_skew0_seed0_image+trajectory_10+10+30 \
    --model mm \
    --algorithm mm_mhd_fedavg \
    --sample uniform \
    --aggregate weighted_scale \
    --num_rounds 500 \
    --proportion 0.3 \
    --num_epochs 2 \
    --learning_rate 0.05 \
    --batch_size 64 \
    --gpu 0 \
    --seed 1234 \
    --test_batch_size 64 \
    --contrastive_weight 1.0 \
    --temperature 0.2