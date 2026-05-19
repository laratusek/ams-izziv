# ams-izziv

## Train

run_training_stunet.py DATASET_NAME_OR_ID CONFIGURATION FOLD -tr TRAINER -pretrained_weights MODEL -p PLAN_ID


## Test

run_test.py -p1 PREDICTIONS_PATH1 -p2 PREDICTIONS_PATH2 -gt GT_PATH -o OUTPUT_PATH 

## Inference

run_inference.py -i INPUT_PATH -o OUTPUT_PATH -d DATASET -tr TRAINER -c CONFIGURATION -chk CHECKPOINT -f FOLD -npp NUM_PROC_PREPROCESS -nps NUM_PROC_SEGMENTATION
