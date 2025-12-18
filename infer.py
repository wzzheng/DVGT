import torch
from dvgt.models.dvgt import DVGT
from dvgt.utils.load_fn import load_and_preprocess_images
from iopath.common.file_io import g_pathmgr

checkpoint_path = 'ckpt/open_ckpt.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
model = DVGT()
with g_pathmgr.open(checkpoint_path, "rb") as f:
    checkpoint = torch.load(f, map_location="cpu")
model.load_state_dict(checkpoint)
model = model.to(device).eval()

# Load and preprocess example images (replace with your own image paths)
image_dir = 'examples/openscene_log-0104-scene-0007'
images = load_and_preprocess_images(image_dir, start_frame=16, end_frame=23).to(device)

with torch.no_grad():
    with torch.amp.autocast(device, dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)