python main.py \
    --task ptbxl_cnum1_iid_0+1 \
    --model respond_kl_one_pretrained \
    --algorithm mm_ptbxl_kl_one_pretrained \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --proportion 1.0 \
    --num_epochs 2 \
    --learning_rate 0.5 \
    --lr_scheduler 0 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --gpu 0 \
    --seed 1234 \
    --test_batch_size 128 \
    --contrastive_weight 0.0 \
    --temperature 0.0 \
    --margin 0.0 \
    --kl_weight 1.0