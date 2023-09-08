import requests
from PIL import Image
import timm
from timm.data.imagenet_info import ImageNetInfo

from semantic_kernel.skill_definition import (
    sk_function,
)
from semantic_kernel.orchestration.sk_context import SKContext


class ImageClassifierPlugin:
    def __init__(self):
        self.model = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True)
        self.model.eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.imagenet_info = ImageNetInfo()

    @sk_function(
        description="Takes a url as an input and classifies the image",
        name="classify_image",
        input_description="The url of the image to classify",
    )
    
    def classify_image(self, url: str) -> str:
        image = self.download_image(url)
        pred = self.model(self.transforms(image)[None])
        cls = self.imagenet_info.index_to_description(pred.argmax())
        
        return cls.split(",")[0]
        

    def download_image(self, url):
        return Image.open(requests.get(url, stream=True).raw).convert("RGB")