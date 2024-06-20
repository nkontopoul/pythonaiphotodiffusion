from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import time
import os

# Logging function
def log(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# Start logging
log("Script started")

# Load the model
log("Loading model...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Ensure the model uses CPU
pipe.to("cpu")
log("Model loaded and moved to CPU")

# Generate an image
log("Generating image...")
prompt = "A ship filled with ancient greek Spartan fighters approaching the harbor of Sparta and people cheering on the port"
start_time = time.time()

try:
    result = pipe(prompt)
    image = result.images[0]
    log(f"Image generated in {time.time() - start_time:.2f} seconds")
except Exception as e:
    log(f"Error during image generation: {e}")
    exit()

# Ensure image is a PIL image
if isinstance(image, Image.Image):
    log("Image generated successfully.")
else:
    log("Generated object is not a PIL Image.")
    exit()

# Save the image
output_path = "output.png"
try:
    image.save(output_path)
    log(f"Image saved successfully at {output_path}")
except Exception as e:
    log(f"Error saving image: {e}")
    exit()

# Check if the image is saved
if os.path.exists(output_path):
    log(f"Image saved successfully at {output_path}")
else:
    log("Failed to save the image.")

# Display the image
try:
    image.show()
    log("Image displayed successfully.")
except Exception as e:
    log(f"Error displaying image: {e}")

log("Script finished")
