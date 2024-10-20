import torch
from config import TrainingConfig
from torchvision.utils import save_image
from diffusers import UNet2DModel, DDPMPipeline, DDPMScheduler
from safetensors import safe_open
from PIL import Image
import os
from train_ridgepatternGAN import RidgePatternGenerator
import torchvision.transforms as transforms
# Path to the model and scheduler checkpoint for fingerprint generation
fingerprint_model_checkpoint_path = "checkpoints/handdorsal_0131_1249.pt"

# Path to the model and scheduler checkpoint for ridge pattern generation
ridge_pattern_model_checkpoint_path = "checkpoints/ckpt_epoch000050.pth"  
# Initialize the fingerprint generation model
fingerprint_checkpoint = torch.load(fingerprint_model_checkpoint_path)
fingerprint_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fingerprint_model = UNet2DModel(
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
).to(fingerprint_device)
fingerprint_model.load_state_dict(fingerprint_checkpoint['model_state_dict'])
fingerprint_noise_scheduler = fingerprint_checkpoint['noise_schedule']
fingerprint_pipeline = DDPMPipeline(unet=fingerprint_model, scheduler=fingerprint_noise_scheduler)

# Initialize the ridge pattern generation model
ridge_pattern_checkpoint = torch.load(ridge_pattern_model_checkpoint_path)
ridge_pattern_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ridge_pattern_model = RidgePatternGenerator()  # Ridge
ridge_pattern_model.load_state_dict(ridge_pattern_checkpoint['modelG_state_dict'])
ridge_pattern_model.to(ridge_pattern_device)
ridge_pattern_model.eval()

# Set the number of sample images to generate and batch size
num_sample_images = 6000
batch_size = 1
output_dir = "handdorsal_with_ridge_pattern"  

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate sample images with fingerprint and ridge pattern
for i in range(num_sample_images):
    with torch.no_grad():
        # Generate a fingerprint image
        fingerprint_images = fingerprint_pipeline(
            batch_size=batch_size,
            generator=torch.manual_seed(i),
        ).images

        # Generate a ridge pattern image from the fingerprint image
        fingerprint_image = fingerprint_images[0]
        ridge_pattern_images = ridge_pattern_model(fingerprint_images).detach().cpu()

    # Save the generated image
    save_image(ridge_pattern_images, os.path.join(output_dir, f"{i:04}.png"))

    print(f"Sample image {i + 1}/{num_sample_images} has been generated and saved.")