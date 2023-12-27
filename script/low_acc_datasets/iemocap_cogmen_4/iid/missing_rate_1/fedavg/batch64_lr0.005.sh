python generate_fedtask.py \
    --benchmark iemocap_cogmen_classification \
    --dist 0 \
    --skew 0.0 \
    --num_clients 20 \
    --seed 0 \
    --missing \
    --modal_equality

python main_iemocap4.py \
    --task iemocap_cogmen_classification_cnum20_dist0_skew0_seed0_mifl_gblend_missing_rate_1 \
    --model fedavg \
    --algorithm multimodal.iemocap_cogmen_classification.fedavg \
    --sample full \
    --aggregate other \
    --num_rounds 1000 \
    --early_stop 40  \
    --proportion 1.0 \
    --lr_scheduler 0 \
    --seed 1234 \
    --fedmsplit_prox_lambda 0.01 \
    --learning_rate 0.005 \
    --num_epochs 3 \
    --learning_rate_decay 1.0 \
    --batch_size 64 \
    --test_batch_size 64 \
    --gpu 1 \
    --wandb