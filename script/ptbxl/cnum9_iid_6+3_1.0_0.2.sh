python main.py \
    --task ptbxl_cnum9_iid_6+3 \
    --model inception1d \
    --algorithm mm_ptbxl \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --proportion 1.0 \
    --num_epochs 5 \
    --learning_rate 0.1 \
    --batch_size 128 \
    --gpu 0 \
    --seed 1234 \
    --test_batch_size 128 \
    --contrastive_weight 1.0 \
    --temperature 0.2