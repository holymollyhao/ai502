python train.py --batch_size 64 --data_parallel --lr 2e-5 --wandb --output_dir clip_adaptive_tune_wzoom_textencoder --cumulative_grad_steps 4 --eval_step 100 --flip_handler zoom --flips "adaptive_flip"
python train.py --batch_size 64 --data_parallel --lr 2e-5 --wandb --output_dir clip_visual_prompt_textencoder --cumulative_grad_steps 4 --eval_step 100 --flip_handler none --flips "adaptive_flip"
