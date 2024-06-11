from PIL import Image, ImageDraw, ImageEnhance
def draw_circle(image: Image, bbox: list, color="red", width=10, center_circle=False):
    image = image.copy()
    if center_circle:
        x0, y0, x1, y1 = bbox
        ctx_x, ctx_y = round((x0+x1)/2), round((y0+y1)/2)
        w, h = x1 - x0, y1 - y0
        radius = round(min(w, h) / 2)
        bbox = [ctx_x - radius, ctx_y - radius, ctx_x + radius, ctx_y + radius]
    image = ImageDraw.Draw(image)
    image.ellipse(bbox, outline=color, width=width)
    return image._image

def center_crop(image: Image, crop_factor: float = 0.5):
    width, height = image.size
    new_width = int(width * crop_factor)
    new_height = int(height * crop_factor)
    new_width, new_height = max(new_height, new_width), max(new_height, new_width)
    x0 = round(width / 2 - new_width / 2)
    y0 = round(height / 2 - new_height / 2)
    x1 = round(width / 2 + new_width / 2)
    y1 = round(height / 2 + new_height / 2)
    return image.crop((x0, y0, x1, y1))

def rotate(image: Image, degree: float, expand:bool=False):
    return image.rotate(degree, expand)

def horizontal_flip(image: Image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip(image: Image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

# Function to scale down the image
def scale_down(image: Image.Image, scale_factor: float = 0.5):
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Function to scale up the image
def scale_up(image: Image.Image, scale_factor: float = 1.5):
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def adjust_color(image: Image.Image, factor = 0.2):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def no_change(image: Image.Image):
    return image

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def zoom_image(image_pil, zoom_factor=1.5):
    """
    Zooms a PIL image by a given zoom factor while preserving the original shape.

    Parameters:
    - image_pil (PIL.Image): Input image as a PIL Image object.
    - zoom_factor (float): Factor by which to zoom the image. Greater than 1 means zooming in, less than 1 means zooming out.

    Returns:
    - zoomed_image_pil (PIL.Image): The zoomed image as a PIL Image object with the original shape.
    """
    # Ensure zoom factor is valid
    assert zoom_factor > 0, "Zoom factor must be greater than 0"

    # Convert PIL image to tensor
    preprocess = transforms.ToTensor()
    image_tensor = preprocess(image_pil)  # Shape: (C, H, W)

    # Get the original dimensions
    _, original_height, original_width = image_tensor.shape

    # Compute new dimensions
    new_height = int(original_height * zoom_factor)
    new_width = int(original_width * zoom_factor)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # Shape: (1, C, H, W)

    # Perform the zoom (interpolation)
    zoomed_image = F.interpolate(image_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # Remove batch dimension
    zoomed_image = zoomed_image.squeeze(0)  # Shape: (C, H, W)

    # Crop or pad the zoomed image to the original size
    if zoom_factor > 1:
        # Crop the center of the zoomed image
        start_y = (new_height - original_height) // 2
        start_x = (new_width - original_width) // 2
        zoomed_image = zoomed_image[:, start_y:start_y + original_height, start_x:start_x + original_width]
    else:
        # Pad the zoomed image to the original size
        pad_y = (original_height - new_height) // 2
        pad_x = (original_width - new_width) // 2
        zoomed_image = F.pad(zoomed_image, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)

    # Convert tensor back to PIL image
    postprocess = transforms.ToPILImage()
    zoomed_image_pil = postprocess(zoomed_image)

    return zoomed_image_pil
