CUDA_VISIBLE_DEVICES=2,3 python scripts/train.py \
--dataset_dir \
"/import/c4dm-datasets/MedleyDB_V1/V1" \
"/import/c4dm-datasets/MedleyDB_V2/V2" \
--dataset_name "MedleyDB" \
--log_dir "/import/c4dm-datasets-ext/automix-toolkit" \
--automix_model "dmc" \
--train_length 262144 \
--val_length 262144 \
--accelerator gpu \
--devices 2 \
--batch_size 8 \
--lr 3e-4 \
--max_epochs 200 \
--max_steps 100000 \
--max_num_tracks 16 \
--num_workers 16 \