import os
import random
import time
import requests
import io
from PIL import Image


class ImageGenerator:
    def __init__(self, config):
        self.config = config

    def generate_image(
        self,
        prompt,
        output_dir=".",
        width=1024,
        height=768,
        num_inference_steps=50,
        guidance_scale=9,
        seed=None,
        scheduler="heunpp2",
    ):
        API_URL = (
            "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
        )
        headers = {"Authorization": f"Bearer {self.config.HF_API_TOKEN}"}

        seed = random.randint(0, 2**32 - 1) if seed is None else seed

        payload = {
            "inputs": prompt,
            "parameters": {
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "scheduler": scheduler,
            },
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Image generation failed: {response.text}")

        image = Image.open(io.BytesIO(response.content))
        timestamp = int(time.time())
        filename = f"image_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)

        os.makedirs(output_dir, exist_ok=True)
        image.save(output_path)

        return image, output_path
