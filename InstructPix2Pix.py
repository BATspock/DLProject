# %%
# !pip install diffusers
# !pip install transformers


# %%
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all'

# %%
import PIL
import requests
import torch
import transformers
# from google.colab import files
from diffusers import StableDiffusionInstructPix2PixPipeline

# %%
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# %%
url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


image = download_image(url)

prompt = "make the mountains snowy"
images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
images[0].save("snowy_mountains.png")

# %%
img = PIL.Image.open("Wiki_training_0226.jpg")
image = PIL.ImageOps.exif_transpose(img)
image = image.convert("RGB")   

# %%
prompt = "make it rainy"
images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
images[0].save("rainy_hollywood.png")


