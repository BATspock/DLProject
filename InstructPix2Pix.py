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
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("mps")


# for all prompts in prompt_list.txt file generate images
with open("prompt_list.txt", "r") as f:
    prompts = f.readlines()
    # remove new line character
    prompts = [prompt.strip() for prompt in prompts]



for prompt in prompts:
    print("#"*50)
    print("Promt:", prompt)

    for i in range(4):
        cur_dalle_img = PIL.Image.open('dalle_api_images/' + prompt + "_" + str(i) + ".jpg")
        for j in range(4):
            cur_img = pipe(prompt, image=cur_dalle_img, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images[0]
            cur_img.save(f"pix2pix_imgs/{prompt}_{i}_{j}.jpg")
    print("#"*50)