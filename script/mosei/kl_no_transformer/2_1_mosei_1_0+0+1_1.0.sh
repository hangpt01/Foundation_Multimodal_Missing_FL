python main.py \
    --task mosei_classification_cnum1_dist0_skew0_seed0_text+vision_0+0+1 \
    --model mm_kl_no_transformer \
    --algorithm mm_mosei_fedavg \
    --sample full \
    --aggregate weighted_scale \
    --num_rounds 500 \
    --proportion 1.0 \
    --num_epochs 2 \
    --learning_rate 0.05 \
    --lr_scheduler 0 \
    --learning_rate_decay 0.9 \
    --batch_size 64 \
    --gpu 0 \
    --seed 1234 \
    --test_batch_size 64 \
    --contrastive_weight 0.0 \
    --temperature 0.0\
    --kl_weight 1.0