CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 mpirun --cpu-set 0-2 --bind-to core -n 1 python entry.py train \
            --conf_files configs/seem/focalt_unicl_lang_v1.yaml \
            --overrides \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 8 \
            TRAIN.BATCH_SIZE_TOTAL 8 \
            TRAIN.BATCH_SIZE_PER_GPU 8 \
            SOLVER.MAX_NUM_EPOCHS 20 \
            SOLVER.BASE_LR 0.0001 \
            ATTENTION_ARCH.QUERY_NUMBER 3 \
            WEIGHT True \
            RESUME_FROM seem_focalt_v1.pt \
            WANDB True
