#!/bin/bash

# run python code
python_script="quantized_inference.py"
echo "Running quantized_inference.py"

nms_thres_values=(0.01)
pt_type=("best_multiscale")

# iterate
# for nn in "${nms_thres_values[@]}"; do
#     # run code and pass through parameter
#     python "$python_script" --nms_threshold "$nn"
# done

# for nn in "${pt_type[@]}"; do
#     python "$python_script" --pt_file_type "$pt_type" --nms_threshold "$nms_thres_values"
# done

python "$python_script" --pt_file_type "$pt_type" --nms_threshold "$nms_thres_values" --pt_file_type "$pt_type"
echo "Job done!"
