CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
--dataset_dir \
"/import/c4dm-datasets/MedleyDB_V1/V1" \
"/import/c4dm-datasets/MedleyDB_V2/V2" \
--dataset_name "MedleyDB" \
--log_dir "/import/c4dm-datasets-ext/automix-toolkit" \
--automix_model "dmc" \
--train_length 131072 \
--val_length 131072 \
--accelerator gpu \
--devices 1 \
--batch_size 16 \
--lr 1e-4 \
--max_epochs 200 \
--max_num_tracks 16 \
--num_workers 2 \
--gradient_clip_val 1.0 \