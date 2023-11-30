python generate_fedtask.py \
    --benchmark iemocap_cogmen_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 20 \
    --seed 0 \
    --missing

python main_iemocap4.py \
    --task iemocap_cogmen_classification_cnum20_dist0_skew0_seed0_mifl_gblend \
    --model mifl \
    --algorithm multimodal.iemocap_cogmen_classification.mifl \
    --sample full \
    --aggregate other \
    --num_rounds 500 \
    --early_stop 40  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.05 \
    --num_epochs 1 \
    --learning_rate_decay 1.0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --gpu 1 
    # \
    # --wandb