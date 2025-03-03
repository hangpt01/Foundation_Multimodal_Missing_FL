python generate_fedtask.py \
    --benchmark food101_baselines \
    --dist 0 \
    --skew 0 \
    --num_clients 20 \
    --seed 0 \
    --missing \
    --missing_ratio_train 0.7 \
    --missing_ratio_test 0.7 \
    --missing_type_train text \
    --missing_type_test text \
    --both_ratio 0 \
    --max_text_len 40
python generate_fedtask.py \
    --benchmark food101_baselines \
    --dist 0 \
    --skew 0 \
    --num_clients 20 \
    --seed 0 \
    --missing \
    --missing_ratio_train 0.7 \
    --missing_ratio_test 0.7 \
    --missing_type_train image \
    --missing_type_test image \
    --both_ratio 0 \
    --max_text_len 40