python notebook/preprocess_food101_3_classes.py

python generate_fedtask.py \
    --benchmark food101_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 10 \
    --seed 0 \
    --missing \
    --num_classes 3
