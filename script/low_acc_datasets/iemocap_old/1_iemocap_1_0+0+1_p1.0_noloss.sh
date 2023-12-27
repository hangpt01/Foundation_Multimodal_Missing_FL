python generate_fedtask.py \
    --benchmark iemocap_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 1 \
    --seed 0 \
    --percentages 0.0 0.0

python main.py \
    --task iemocap_classification_cnum1_dist0_skew0_seed0_text+audio+vision_0+0+1 \
    --model mm_clkl_separate3 \
    --algorithm mm_iemocap_fedavg_clkl_separate3 \
    --sample uniform \
    --aggregate weighted_scale \
    --num_rounds 500 \
    --proportion 1.0 \
    --num_epochs 2 \
    --learning_rate 0.0001 \
    --lr_scheduler 0 \
    --learning_rate_decay 0.1 \
    --batch_size 32 \
    --gpu 0 \
    --seed 1234 \
    --test_batch_size 32  \
    --contrastive_weight 0.0 \
    --temperature 1.0\
    --kl_weight 0.0