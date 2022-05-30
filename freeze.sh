python train.py --train_dataset_dir="/home/dataset/augmented_data" \
                --val_dataset_dir="/home/dataset/generated_data" \
                --logdir="./log/Enet" \
                --is_training=False \
                --num_classes=2 \
                --batch_size=1 \
                --eval_batch_size=25 \
                --image_height=160 \
                --image_width=160 \
                --weighting="ENET" \
                --num_initial_blocks=1 \
                --stage_two_repeat=1 \
                --skip_connections=False

python genpb.py --ckpt_dir="./log/Enet"


