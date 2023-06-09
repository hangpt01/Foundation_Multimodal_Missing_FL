export PYTHONPATH='.'
export CUDA_VISIBLE_DEVICES='0'
python centralize_test/main_vae.py \
    --lr 0.005 \
    --lr_scheduler_type by_step \
    --lr_decay_rate 1.0 \
    --step_size 2 \
    --batch_size 64 \
    --epochs 100 \
    --seed 1234 \
    --modalities image sound \
    --wandb