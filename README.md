# HiFT on Horizon X3 pi

This is an unofficial X3 pi deployment of HiFT[1] based on Python, whose official verision is as follows:  
[Official implementation of HiFT](https://github.com/vision4robotics/HiFT)

## Model Overview
Tracker | MACs | Params | FPS | Avg Latency | DDR Latency | Subgraph | BPU Util1 | BPU Util2 | DTB70 Success
--- | --- | --- | --- |--- |--- |--- |--- |--- |---
HiFT | 4.83G | 9.98M | 4.215 | 941.55ms | 884.52ms | 33 | 19.29% | 19.94% | 0.594

We test BPU utilization rate by using hrut_somstatus while testing static performance with 4 threads:
```
hrut_somstatus -n 10000 -d 1
```
We record the outputs of hrut_somstatus, delete the values which are lower than 10% 
and compute the average of BPU utilization rate.  
Note that in order to ensure fairness when compared with other trackers, 
we do not execute any optimization of operators when deploying model on Horizon X3 pi.

## Demos for ChasingDrones
```
cd video
tar zxvf ChasingDrones.tar.gz
```
<div align="center">
  <img src="https://github.com/STQ-AmadeusUser/HiFT-X3/blob/main/images/ChasingDrones_result.gif">
</div>
The GIF is generated by using moviepy to covert ChasingDrones_result.mp4, which the demo on X3 pi outputs:

```
clip = (VideoFileClip("ChasingDrones_result.mp4").resize((640, 360)))
clip.write_gif("ChasingDrones_result.gif", fps=15)
```
### A Demo on X3 pi
You can execute running_on_X3.py to experience HiFT on ChasingDrones, one video in DTB70[2] benchmark:
```
cd tools
python demo_X3.py
```
The code will generate a video named ChasingDrones_result.mp4 in "video" directory.

### A Demo on Laptop
We also provide demo.py for laptop running (PyTorch required):
```
cd tools
python demo.py
```
The code will generate a video named ChasingDrones_result.mp4 in "video" directory.

## How to Deploy (for developers)
We have converted the network of HiFT to HiFT.bin, a file with X3-pi-specialized format, in "bin" directory.  
If you want to generate the file by yourself, you can take advantage of our code and [follow this instruction](https://developer.horizon.cc/documents_rdk/category/toolchain_development).
1. Get the onnx model: HiFT.onnx
```
cd deploy
python convert_onnx.py
```
You can visualize onnx model by using [netron](https://netron.app/).

2. Simplify onnx model: HiFT_sim.onnx
```
python -m onnxsim HiFT.onnx HiFT_sim.onnx
```
3. Generate files for calibration:
```
python calibration.py
```
This step requires training datasets described in official implementation of HiFT or pysot[3].
If you do not want to prepare datasets, we also provide calibration files in calibration.tar.gz.

4. Convert onnx model: HiFT.bin
```
hb_mapper makertbin --config HiFT.yaml --model-type onnx
```
This step needs Horizon AI Tool-chain. We create an anaconda environment for python packages 
as mentioned in [here](https://developer.horizon.cc/documents_rdk/toolchain_development/beginner).
Note that we do not use docker image. HiFT.yaml contains all parameters used in conversion.

5. Log file is "deploy/hb_mapper_makertbin.log".
6. Other generated files are in "deploy/model" directory.

## Static Performance
We provide static performance of HiFT by running on X3 pi:
1. 1 thread:
```
hrt_model_exec perf --model_file HiFT.bin --thread_num 1
```
<div align="center">
  <img src="https://github.com/STQ-AmadeusUser/HiFT-X3/blob/main/images/1_thread.png">
</div>

2. 4 thread:
```
hrt_model_exec perf --model_file HiFT.bin --thread_num 4
```
<div align="center">
  <img src="https://github.com/STQ-AmadeusUser/HiFT-X3/blob/main/images/4_thread.png">
</div>

## Version
1. System on X3 pi: [ubuntu-preinstalled-server-arm64.img.xz](http://sunrise.horizon.cc/downloads/os_images/2.0.0/release/) 
   (2.0.0)
2. Horizon AI Tool-chain:
    ```
    wget -c ftp://xj3ftp@vrftp.horizon.ai/ai_toolchain/ai_toolchain.tar.gz --ftp-password=xj3ftp@123$%
    ```
    1. hbdk-3.48.6-cp38-cp38-linux_x86_64.whl
    2. hbdk_model_verifier-3.48.6-py3-none-linux_x86_64.whl
    3. horizon_nn-0.21.2-cp38-cp38-linux_x86_64.whl
    4. horizon_tc_ui-1.21.6-cp38-cp38-linux_x86_64.whl
3. [OE package](https://developer.horizon.ai/forumDetail/136488103547258769): horizon_xj3_open_explorer_v2.6.2b-py38_20230606
    ```
    wget -c ftp://vrftp.horizon.ai/Open_Explorer_gcc_9.3.0/2.6.2b/horizon_xj3_open_explorer_v2.6.2b-py38_20230606.tar.gz
    ```
4. PyTorch:
    1. for convert_onnx.py: torch-1.8.0+cu111-cp37-cp37m-linux_x86_64.whl
    2. for demo on laptop: torch-1.9.0+cu111-cp37-cp37m-linux_x86_64.whl

## Reference
[1] HiFT: Hierarchical Feature Transformer for Aerial Tracking  
[2] DTB70: https://github.com/flyers/drone-tracking  
[3] PySOT: https://github.com/STVIR/pysot
