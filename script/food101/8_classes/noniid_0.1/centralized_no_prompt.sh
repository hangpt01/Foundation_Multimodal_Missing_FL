python generate_fedtask.py \
    --benchmark food101_classification_8_classes \
    --dist 1 \
    --skew 0.1 \
    --num_clients 1 \
    --seed 0 \
    --missing \
    --missing_ratio_train 0.7 \
    --missing_ratio_test 0.7 \

python main.py \
    --task food101_classification_8_classes_cnum1_dist1_skew0.1_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
    --model centralized_no_prompt \
    --algorithm multimodal.food101_classification_8_classes.fedavg_no_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 50  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.01 \
    --num_epochs 1 \
    --learning_rate_decay 1 \
    --batch_size 64 \
    --test_batch_size 64 \
    --gpu 0 \
    --wandb