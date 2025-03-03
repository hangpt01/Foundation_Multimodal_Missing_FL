python generate_fedtask.py \
    --benchmark imdb_classification \
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
    --max_text_len 128
python main.py \
    --task imdb_classification_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_text_text_both_ratio_0.0 \
    --model Prob_prompt \
    --algorithm multimodal.imdb_classification.Prob_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 250 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 5678 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.01 \
    --num_epochs 1 \
    --num_outer_loops 5 \
    --learning_rate_decay 1.0 \
    --note loadvilt \
    --batch_size 256 \
    --test_batch_size 256 \
    --max_text_len 128 \
    --gpu 0 \
    --wandb
python main.py \
    --task imdb_classification_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_text_text_both_ratio_0.0 \
    --model L2P \
    --algorithm multimodal.imdb_classification.L2P \
    --sample full \
    --aggregate other \
    --num_rounds 250 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 5678 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.01 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --note loadvilt \
    --batch_size 256 \
    --test_batch_size 256 \
    --max_text_len 128 \
    --gpu 0 \
    --wandb