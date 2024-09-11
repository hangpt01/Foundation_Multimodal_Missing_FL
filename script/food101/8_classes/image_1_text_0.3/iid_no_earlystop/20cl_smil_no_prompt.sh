python main.py \
    --task food101_classification_8_classes_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_text_text_both_ratio_0.0 \
    --model smil_no_prompt \
    --algorithm multimodal.food101_classification_8_classes.smil_no_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 4 \
    --test_batch_size 4 \
    --gpu 1 
    # \
    # --wandb
