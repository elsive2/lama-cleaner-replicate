import replicate
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import InpaintRequest
from PIL import Image
import numpy as np
import io

class Predictor:
    def __init__(self):
        self.model = ModelManager(
            name="lama",  # можно поменять на "zits", "mat", "fcf", "paint-by-example" и т.д.
            device="cpu"
        )

    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image_np = np.array(image.convert("RGB"))
        mask_np = np.array(mask.convert("L"))

        req = InpaintRequest(
            image=image_np,
            mask=mask_np,
            size_limit=2048,
            box=[0, 0, image_np.shape[1], image_np.shape[0]],
            hd_strategy="Original",
            hd_strategy_crop_margin=0,
            hd_strategy_crop_trigger_size=0,
            hd_strategy_resize_limit=0,
            prompt="",
            negative_prompt="",
            use_cf_guidance=False,
            cf_guidance_steps=0,
            cf_guidance_scale=1.0,
        )

        result = self.model(req)
        result_img = Image.fromarray(result)
        return result_img