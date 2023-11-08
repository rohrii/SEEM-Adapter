NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=1 mpirun -n 1 python entry.py evaluate \
            --conf_files configs/seem/focalt_unicl_lang_v1.yaml \
            --overrides \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 4 \
            TRAIN.BATCH_SIZE_TOTAL 4 \
            TRAIN.BATCH_SIZE_PER_GPU 4 \
            SOLVER.MAX_NUM_EPOCHS 1 \
            SOLVER.BASE_LR 0.0001 \
            ATTENTION_ARCH.QUERY_NUMBER 3 \
            WEIGHT True \
            WANDB False \
            RESUME_FROM seem_focalt_v1.pt \
