from config import TrainingConfig
from datasets import load_dataset
import os
from torchvision import transforms
import torch
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
import math
from accelerate import Accelerator, notebook_launcher
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
# Load training configuration
config = TrainingConfig()

# Load the dataset
dataset = load_dataset("imagefolder", data_dir="dataset/Fingerprint_Optical/Fingerprint_Optical", split="train")

# Image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# Create the model
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)


gpu_ids = [0] 
device = torch.device("cuda")
#if torch.cuda.is_available() and len(gpu_ids) > 1:
#    print("CUDA is available", "len gpu_ids", len(gpu_ids))
#    device = torch.device("cuda")
#    try:
#        
#        if __name__ == "__main__":
#            #torch.multiprocessing.set_start_method('spawn')
#            model = nn.DataParallel(model, device_ids=gpu_ids)
#    except Exception as e:
#        print('Exception!', str(e))
#        exit(1357)

model = model.to(device)

# Noise scheduler configuration
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

# Function to create a grid of images
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

# Function to evaluate the model and save images
def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=8, cols=8)

    # Save the images
    test_dir = os.path.join(config.output_dir, "logs")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/fingerprint_KISA_diffusion_{epoch:04d}.png")

# Function to get the full repository name
def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

# Training loop
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
            repo.git_pull()
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process and (epoch + 1) % 20 == 0:
            if isinstance(model, nn.DataParallel):
                model_save_path = os.path.join(config.output_dir, f"fingerprint_KISA_{epoch}.pt")
               # model_and_schedule = {
               #     'model_state_dict': model.module.state_dict(),
               #     'noise_schedule': noise_scheduler,
               #     'optimizer_state_dict': optimizer.state_dict(),
               #       }
               # torch.save(model_and_schedule, model_save_path)
            else:
                model_save_path = os.path.join(config.output_dir, f"fingerprint_KISA_{epoch}.pt")
                model_and_schedule = {
                    'model_state_dict': model.state_dict(),
                    'noise_schedule': noise_scheduler,
                    'optimizer_state_dict': optimizer.state_dict(),
                      }
                torch.save(model_and_schedule, model_save_path)
         #After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            #if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                #if config.push_to_hub:
                    #repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                    #print("push_to_hub")
                    
                    #pipeline.save_pretrained(config.output_dir)
                    #print("===============save model==============")

# Args for training loop
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

# Launch training loop
notebook_launcher(train_loop, args, num_processes=1)