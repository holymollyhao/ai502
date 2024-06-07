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

def adjust_color(image: Image.Image, factor: float):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)