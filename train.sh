python train.py --train_dataset_dir="/home/shubhamjoshi/Downloads/Human-Segmentation-Dataset-master/dataset" \
                --val_dataset_dir="/home/shubhamjoshi/Downloads/Human-Segmentation-Dataset-master/dataset" \
                --logdir="./log/Enet" \
                --is_training=True \
                --save_images=True \
                --combine_dataset=False \
                --num_classes=2 \
                --batch_size=10 \
                --eval_batch_size=25 \
                --image_height=160 \
                --image_width=160 \
                --num_epochs=300 \
                --num_epochs_before_decay=100 \
                --weighting="ENET" \
                --num_initial_blocks=1 \
                --stage_two_repeat=1 \
                --skip_connections=False


