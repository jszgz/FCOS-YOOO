#tensorboard
tensorboard --logdir runs --port=11990

#train

export PYTHONPATH=/home/wt/PycharmProjects/FCOS:$PYTHONPATH

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x


#train YOOO_FIRST

nohup python -u tools/train_net.py  --config-file configs/fcos/fcos_imprv_R_50_FPN_1x_singleTITAN_Train_NCAA_YOOO.yaml DATALOADER.NUM_WORKERS 2 OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x_singleTITAN_Train_NCAA_YOOO_WITHTSM_CROSSENTROPY_01234  > fcos_imprv_R_50_FPN_1x_singleTITAN_Train_NCAA_YOOO_WITHTSM_CROSSENTROPY_01234.txt 2>&1 &
