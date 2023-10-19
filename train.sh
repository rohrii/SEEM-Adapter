CUDA_VISIBLE_DEVICES=0 mpirun -n 1 python entry.py train \
            --conf_files configs/seem/focalt_unicl_lang_v1.yaml \
            --overrides \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 4 \
            TRAIN.BATCH_SIZE_TOTAL 4 \
            TRAIN.BATCH_SIZE_PER_GPU 4 \
            SOLVER.MAX_NUM_EPOCHS 1 \
            SOLVER.BASE_LR 0.0001 \
            SOLVER.FIX_PARAM.backbone True \
            SOLVER.FIX_PARAM.lang_encoder True \
            SOLVER.FIX_PARAM.pixel_decoder True \
            ATTENTION_ARCH.QUERY_NUMBER 3 \
            WEIGHT True \
            RESUME_FROM xdecoder_focalt_last.pt