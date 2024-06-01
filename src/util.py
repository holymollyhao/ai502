from PIL import Image

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
