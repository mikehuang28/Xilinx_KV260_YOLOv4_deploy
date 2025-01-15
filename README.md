# yolov4_csp deploy to KV260
This is the yolov4_csp model deployment onto KV260.

For more details, please visit "Model Deployment on KV260 iVS Lab User Guide"

## Quantization
### Run Quantization
``````
[Docker] python3 test.py --cfg ./cfg/GM_E_shortcut_prune_ratio_0.9_P85_yolov4_CSP_672_2.cfg --data data/coco_3cls.yaml --weight ./weights/best_2.pt --quant_mode calib --output_path /workspace/quantize_result --nndct_quant --img-size 640
``````

### Generate .xmodel File for Compilation
``````
[Docker] python3 test.py --cfg ./cfg/GM_E_shortcut_prune_ratio_0.9_P85_yolov4_CSP_672_2.cfg --data data/coco_3cls.yaml --weight ./weights/best_2.pt --dump_xmodel --output_path /workspace/quantize_result/ --nndct_quant --quant_mode test --batch-size 1 --deploy --img-size 640
``````

## Quantized Model Inference (Optional)

``````
[Docker] ./run_quantized_inference.sh
``````

## Compilation
``````
[Docker] $ vai_c_xir -x /quantize_result/Darknet_0_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o yolov4_csp_pt -n yolov4_csp_pt
``````
## Reference
* [Vitis AI user guide](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai)
* [Vitis AI library user guide](https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk/Introduction)
* [Quick Start Guide for Zynq™ UltraScale+™](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html#quickstart)
* [Yolov4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [Vitis AI copyleft model zoo](https://github.com/Xilinx/Vitis-AI-Copyleft-Model-Zoo/tree/main)
* [AMD FPGA support](https://support.xilinx.com/s/topic/0TO2E000000YKY9WAO/vitis-ai-ai?language=en_US)