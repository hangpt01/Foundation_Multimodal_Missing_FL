python generate_fedtask.py \
    --benchmark food101_classification_8_classes \
    --dist 1 \
    --skew 0.5 \
    --num_clients 20 \
    --seed 0 \
    --missing \
    --missing_ratio_train 0.7 \
    --missing_ratio_test 0.7 \