python main.py \
    --task hatememe_classification_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_text_text_both_ratio_0.0 \
    --model L2P_Prob_prompt \
    --algorithm multimodal.hatememe_classification.L2P_Prob_prompt \
    --sample full \
    --aggregate other \
    --num_rounds 300 \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0 \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --num_outer_loops 5 \
    --learning_rate_decay 1.0 \
    --note size_20_topk_5x2 \
    --batch_size 512 \
    --test_batch_size 512 \
    --gpu 0 \
    --wandb