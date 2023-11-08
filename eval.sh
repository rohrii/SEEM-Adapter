NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=1 mpirun --cpu-set 0-9 --bind-to core -n 1 python entry.py evaluate \
            --conf_files configs/seem/focalt_unicl_lang_v1.yaml \
            --overrides \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 8 \
            ATTENTION_ARCH.QUERY_NUMBER 3 \
            WEIGHT True \
            WANDB False \
            RESUME_FROM seem_focalt_v1.pt \
