CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
--dataset_dir "/import/c4dm-datasets/ENST-drums" \
--dataset_name "enst-drums" \
--log_dir "/import/c4dm-datasets-ext/automix-toolkit" \
--automix_model "dmc" \
--train_length 262144 \
--val_length 262144 \
--accelerator gpu \
--devices 1 \
--max_epochs 100 \
--batch_size 4 \
--lr 1e-5 \

