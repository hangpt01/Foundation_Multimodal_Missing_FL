python generate_fedtask.py \
    --benchmark food101_classification_8_classes \
    --dist 0 \
    --skew 0 \
    --num_clients 20 \
    --seed 0 \
    --missing \
    --missing_ratio_train 0.7 \
    --missing_ratio_test 0.7 \
    --missing_type_train both \
    --missing_type_test both \
    --both_ratio 0.5 \
    --max_text_len 40
python main.py \
    --task food101_classification_8_classes_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
    --model L2P \
    --algorithm multimodal.food101_classification_8_classes.L2P \
    --sample full \
    --aggregate other \
    --num_rounds 250 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 91011 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --note loadvilt \
    --batch_size 512 \
    --test_batch_size 512 \
    --max_text_len 40 \
    --gpu 0 \
    --wandb
python main.py \
    --task food101_classification_8_classes_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 \
    --model Prob_prompt \
    --algorithm multimodal.food101_classification_8_classes.Prob_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 250 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 91011 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --num_outer_loops 5 \
    --learning_rate_decay 1.0 \
    --note loadvilt \
    --batch_size 512 \
    --test_batch_size 512 \
    --max_text_len 40 \
    --gpu 0 \
    --wandb