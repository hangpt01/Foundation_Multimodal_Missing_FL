python generate_fedtask.py \
    --benchmark mnist_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 3 \
    --seed 0

python main.py \
    --task mnist_classification_cnum3_dist0_skew0_seed0 \
    --model mlp \
    --algorithm fedavg \
    --sample full \
    --aggregate other \
    --num_rounds 1 \
    --proportion 1.0 \
    --num_epochs 1 \
    --learning_rate 0.01 \
    --batch_size 32 \
    --gpu 0 \
    --seed 1234 \
    --test_batch_size 32