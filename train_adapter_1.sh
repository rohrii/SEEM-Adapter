CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 mpirun --cpu-set 0-4 --bind-to core -n 1 python entry.py train \
    --conf_files configs/seem/focall_unicl_lang_v1.yaml \
    --overrides \
    MODEL.DECODER.HIDDEN_DIM 512 \
    MODEL.ENCODER.CONVS_DIM 512 \
    MODEL.ENCODER.MASK_DIM 512 \
    MODEL.ENCODER.NUM_CLASSES 12 \
    DATASETS.TRAIN '["xray-waste-train"]' \
    DATASETS.TEST '["xray-waste-val"]' \
    TEST.BATCH_SIZE_TOTAL 4 \
    TRAIN.BATCH_SIZE_TOTAL 4 \
    TRAIN.BATCH_SIZE_PER_GPU 4 \
    SOLVER.MAX_NUM_EPOCHS 15 \
    ATTENTION_ARCH.QUERY_NUMBER 3 \
    WEIGHT True \
    RESUME_FROM seem_focall_v1.pt \
    WANDB True \
    SOLVER.IGNORE_FIX '["class_embed", "mask_embed", "pixel_decoder_self_attention_adapter", "decoder_self_attention_adapter", "decoder_cross_attention_adapter"]' \
    USE_ADAPTERS True \
    ADAPTER_NUM 1
