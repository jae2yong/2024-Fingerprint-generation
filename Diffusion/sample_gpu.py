import torch
from config import TrainingConfig
from torchvision.utils import save_image
from diffusers import UNet2DModel, DDPMPipeline, DDPMScheduler
from safetensors import safe_open
from PIL import Image
import os
# Path to the model and scheduler checkpoint
model_checkpoint_path = "fingerprint_optical_3449.pt"

# Initialize the model and scheduler, and load their states
checkpoint = torch.load(model_checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
loaded_model = UNet2DModel(
    #sample_size=config.image_size,
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)
 
# Load the model's state dictionary
loaded_model.load_state_dict(checkpoint['model_state_dict'])

# Load the scheduler
loaded_noise_scheduler = checkpoint['noise_schedule']

# Initialize the DDPMPipeline
pipeline = DDPMPipeline(unet=loaded_model, scheduler=loaded_noise_scheduler)

# Set the number of sample images to generate and batch size
num_sample_images = 1
batch_size = 1
output_dir = "fingerprint_optical"
for i in range(3000):
  # Generate sample images
  generated_images = pipeline(
      batch_size=batch_size,
      generator=torch.manual_seed(i),
  ).images
  
  # Create a grid of images
  def make_grid(images, rows, cols):
      w, h = images[0].size
      grid = Image.new("RGB", size=(cols * w, rows * h))
      for i, image in enumerate(images):
          grid.paste(image, box=(i % cols * w, i // cols * h))
      return grid
  
  # Create an image grid
  image_grid = make_grid(generated_images, rows=1, cols=1)
  filename = os.path.join(output_dir, f"{i:04}.png")
  # Save the image grid
  image_grid.save(filename)
  
  print("Sample images have been generated and saved to 'sample_generated_images.py'")