CUDA_VISIBLE_DEVICES=0,1 mpirun --cpu-set 0-12 --bind-to core -n 2 python entry.py train \
    --conf_files configs/seem/focall_unicl_lang_v1.yaml \
    --overrides \
    WEIGHT seem_focall_v1.pt \
    MODEL.ENCODER.NUM_CLASSES <<NUM CLASSES>> \
    TEST.BATCH_SIZE_TOTAL 4 \
    TRAIN.BATCH_SIZE_TOTAL 4 \
    TRAIN.BATCH_SIZE_PER_GPU 2 \
    DATASETS.TRAIN "['<<DATASET NAME TRAIN>>']" \
    DATASETS.TEST "['<<DATASET NAME VAL>>']" \
    INPUT.PIXEL_MEAN "[92.2797, 154.551, 153.192]" \
    INPUT.PIXEL_STD "[67.1433, 39.6286, 41.3266]" \
    SOLVER.MAX_NUM_EPOCHS <<NUM EPOCHS>> \
    SOLVER.BASE_LR 0.001 \
    SOLVER.STEPS "[0.88889, 0.96296]" \
    SOLVER.IGNORE_FIX "['class_embed', 'mask_embed']" \
    USE_LORA True \
    LORA_TARGETS "['q','v']" \
    LORA_RANK 8 \
    LORA_ALPHA 8 \
    WANDB True \
    WANDB_EXP_NAME "<<WANDB EXPERIMENT NAME>>"
