from PIL import Image

def rotate(image: Image, degree: float, expand:bool=False):
    return image.rotate(degree, expand)

def horizontal_flip(image: Image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip(image: Image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)