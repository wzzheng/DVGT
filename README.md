# DVGT: Driving Visual Geometry Transformer
**DVGT** is a comprehensive autonomous driving framework that leverages dense 3D geometry as the foundation for perception and planning. This repository hosts the official implementation of the DVGT series: from reconstructing metric-scaled dense point maps across diverse datasets (**DVGT-1**) to introducing an efficient Vision-Geometry-Action (VGA) paradigm for online joint reconstruction and planning (**DVGT-2**). 

***Check our project pages ([DVGT-1](https://wzzheng.net/DVGT/), [DVGT-2](https://wzzheng.net/DVGT-2/)) for full demo videos and interactive results!***

## 🚀 DVGT-2 Demos

| Demonstration | Highlight & Description |
| :--- | :--- |
| <img src="./assets/demo-1.gif" width="500"> | **Dense Scene Representation**<br><br>Unlike models relying on inverse perspective mapping or sparse perception results, **DVGT-2** reconstructs dense 3D geometry to provide a comprehensive and detailed scene representation. |
| <img src="./assets/demo-2.gif" width="500"> | **Streaming Reconstruction & Planning**<br><br>Given unposed multi-view image sequences, **DVGT-2** performs joint geometry reconstruction and trajectory planning in a fully online manner for continuous and robust driving. |
| <img src="./assets/demo-3.gif" width="500"> | **Global Geometry Consistency**<br><br>Operating on online input sequences, **DVGT-2** streamingly infers the global geometry of the entire scene, demonstrating high fidelity and temporal consistency. |


## 📖 Publications
### DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale
<div align="center">
    <a href="https://arxiv.org/abs/2604.00813"><img src="https://img.shields.io/badge/arXiv-2604.00813-b31b1b" alt="arXiv"></a>
    <a href='https://wzzheng.net/DVGT-2/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
    <a href='https://huggingface.co/RainyNight/DVGT-2'><img src='https://img.shields.io/badge/🤗%20Hugging%20Face-DVGT-ffd21e'></a>
</div>

**DVGT-2** introduces an efficient Vision-Geometry-Action (VGA) paradigm for autonomous driving. By processing multi-view inputs in an online manner, it jointly performs dense 3D geometry reconstruction and trajectory planning, achieving superior 3D perception and planning capabilities with high efficiency.
<p align="center">
  <img src="./assets/teaser-2.png" width="100%">
</p>

### DVGT-1: Driving Visual Geometry Transformer
<div align="center">
    <a href="https://arxiv.org/abs/2512.16919"><img src="https://img.shields.io/badge/arXiv-2512.16919-b31b1b" alt="arXiv"></a>
    <a href='https://wzzheng.net/DVGT/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
    <a href='https://huggingface.co/RainyNight/DVGT-1'><img src='https://img.shields.io/badge/🤗%20Hugging%20Face-DVGT-ffd21e'></a>
</div>

**DVGT-1** is a universal driving visual geometry model that directly reconstructs metric-scaled dense 3D point maps from unposed multi-view images. It demonstrates robust performance and remarkable generalizability across diverse camera setups and driving scenarios, eliminating the need for post-alignment with external data.
<p align="center">
  <img src="./assets/teaser-1.png" width="100%">
</p>

## ✨ News
- **[2026/04/01]** **DVGT-2:** Paper released.
- **[2026/03/31]** **DVGT-1 & DVGT-2:** Training, evaluation, and data annotation code released.
- **[2026/02/24]** **DVGT-1:** 🎉 🎉 Accepted to CVPR 2026!
- **[2025/12/19]** **DVGT-1:** Paper, inference code, and visualization checkpoints released.

## 📦 Installation

We tested the code with CUDA 12.8, python3.11 and torch 2.8.0.
```bash
git clone https://github.com/wzzheng/DVGT.git
cd dvgt

conda create -n dvgt python=3.11
conda activate dvgt

pip install -r requirements.txt

cd third_party/
git clone https://github.com/facebookresearch/dinov3.git
```
## 🤗 Pretrained Models
Our pretrained models are available on the huggingface hub:

<table>
  <thead>
    <tr>
      <th>Version</th>
      <th>Hugging Face Model</th>
      <th>Metric scale</th>
      <th>Streaming</th>
      <th>#Params</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DVGT-1</td>
      <td><a href="https://huggingface.co/RainyNight/DVGT-1" target="_blank"><code>RainyNight/DVGT-1</code><a></td>
      <td>✅</td>
      <td>-</td>
      <td>1.7B</td>
    </tr>
    <tr>
      <td >DVGT-2</td>
      <td><a href="https://huggingface.co/RainyNight/DVGT-2" target="_blank"><code>RainyNight/DVGT-2</code></a></td>
      <td>✅</td>
      <td>✅</td>
      <td>1.8B</td>
    </tr>
  </tbody>
</table>

## 💡 Minimal Code Example 

Now, try the model with just a few lines of code:

```python
import torch
from dvgt.models.architectures.dvgt1 import DVGT1
# from dvgt.models.architectures.dvgt2 import DVGT2
from dvgt.utils.load_fn import load_and_preprocess_images
from iopath.common.file_io import g_pathmgr

checkpoint_path = 'ckpt/dvgt1.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
model = DVGT1()
# model = DVGT2() # Let'2 try DVGT2
with g_pathmgr.open(checkpoint_path, "rb") as f:
    checkpoint = torch.load(f, map_location="cpu")
model.load_state_dict(checkpoint)
model = model.to(device).eval()

# Load and preprocess example images (replace with your own image paths)
image_dir = 'visual_demo_examples/openscene_log-0104-scene-0007'
images = load_and_preprocess_images(image_dir).to(device)

with torch.no_grad():
    with torch.amp.autocast(device, dtype=dtype):
        # Predict attributes including ego pose and point maps.
        predictions = model(images)
```

## 💡 Detailed Usage

<details>
<summary>Click to expand</summary>

You can also optionally choose which attributes (branches) to predict, as shown below. This achieves the same result as the example above. This example uses a batch size of 1 (processing a single scene), but it naturally works for multiple scenes.

```python
import torch
from dvgt.models.architectures.dvgt1 import DVGT1
# from dvgt.models.architectures.dvgt2 import DVGT2
from dvgt.utils.load_fn import load_and_preprocess_images
from iopath.common.file_io import g_pathmgr
from dvgt.utils.pose_encoding import decode_pose
from dvgt.evaluation.utils.geometry import convert_point_in_ego_0_to_ray_depth_in_ego_n

checkpoint_path = 'ckpt/dvgt1.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
model = DVGT1()
# model = DVGT2() # Let'2 try DVGT2
with g_pathmgr.open(checkpoint_path, "rb") as f:
    checkpoint = torch.load(f, map_location="cpu")
model.load_state_dict(checkpoint)
model = model.to(device).eval()

# Load and preprocess example images (replace with your own image paths)
image_dir = 'visual_demo_examples/openscene_log-0104-scene-0007'
images = load_and_preprocess_images(image_dir).to(device)

with torch.no_grad():
    with torch.amp.autocast(device, dtype=dtype):
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict ego n to ego first
    pose_enc = model.ego_pose_head(aggregated_tokens_list)[-1]
    # Ego pose following the OpenCV convention, relative to the ego-frame of the first time step.
    ego_n_to_ego_0, _ = decode_pose(pose_enc)

    # Predict Point Maps in the ego-frame of the first time step
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

    # The predicted ray depth maps are originated from each ego-vehicle's position in its corresponding frame.
    ray_depth_in_ego_n = convert_point_in_ego_0_to_ray_depth_in_ego_n(point_map, ego_n_to_ego_0)    
```
</details>

### Visualization

Run the following command to perform reconstruction and visualize the point clouds in Viser. This script requires a path to an image folder formatted as follows:

    data_dir/
    ├── frame_0/ (contains view images, e.g., CAM_F.jpg, CAM_B.jpg...)
    ├── frame_1/
    ...

**Note on Data Requirements:**
1. **Consistency:** The data must be sampled at **2Hz**. All frames must contain the same number of views arranged in a fixed order.
2. **Capacity:** DVGT1 inference supports up to 24 frames, while DVGT2 supports arbitrary frame lengths. Both models support an arbitrary number and order of views per frame.

You can directly download our example dataset to get started:
```bash
# around 80MB
wget https://huggingface.co/datasets/RainyNight/DVGT_demo_dataset/resolve/main/visual_demo_examples.zip
unzip visual_demo_examples.zip
```

```bash
python demo_viser.py \
  --model_name=DVGT1 \
  --image_folder=visual_demo_examples/openscene_log-0104-scene-0007
```

## 🌟 Data preparation

See [docs/data_preparation.md](docs/data_preparation.md)

## 🏋️‍♂️ Training & Finetuning

See [docs/train.md](docs/train.md)

## 🧪 Evaluation

See [docs/eval.md](docs/eval.md)

## 🌋 visualization

See [docs/visualization.md](docs/visualization.md)

## Acknowledgements
Our code is based on the following brilliant repositories:

[Moge-2](https://github.com/microsoft/MoGe) 
[CUT3R](https://github.com/CUT3R/CUT3R) 
[Driv3R](https://github.com/Barrybarry-Smith/Driv3R) 
[VGGT](https://github.com/facebookresearch/vggt) 
[MapAnything](https://github.com/facebookresearch/map-anything) 
[Pi3](https://github.com/yyfz/Pi3)

Many thanks to these authors!

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{zuo2025dvgt,
  title={DVGT: Driving Visual Geometry Transformer}, 
  author={Zuo, Sicheng and Xie, Zixun and Zheng, Wenzhao and Xu, Shaoqing and Li, Fang and Jiang, Shengyin and Chen, Long and Yang, Zhi-Xin and Lu, Jiwen},
  journal={arXiv preprint arXiv:2512.16919},
  year={2025}
}

@article{zuo2026dvgt-2,
  title={DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale}, 
  author={Zuo, Sicheng and Xie, Zixun and Zheng, Wenzhao and Xu, Shaoqing and Li, Fang and Li, Hanbing and Chen, Long and Yang, Zhi-Xin and Lu, Jiwen},
  journal={arXiv preprint arXiv:2604.00813},
  year={2026}
}
```

## License

The code and pretrained weights of DVGT-2 are released under the Apache License 2.0.
