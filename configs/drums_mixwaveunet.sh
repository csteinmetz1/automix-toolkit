CUDA_VISIBLE_DEVICES=2,3 python scripts/train.py \
--dataset_dir "/import/c4dm-datasets/ENST-drums" \
--dataset_name "enst-drums" \
--default_root_dir "/import/c4dm-datasets/automix-toolkit" \
--automix_model "simple-waveunet" \
--train_length 262144 \
--val_length 262144 \
--accelerator gpu \
--devices 2 \
--max_epochs 100 \
--batch_size 4 \

