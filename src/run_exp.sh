python train.py --batch_size 64 --data_parallel --lr 2e-5 --wandb --output_dir clip_raaug --cumulative_grad_steps 4 --eval_step 100 --flip_handler zoom --flips "relation_aware_flip"
