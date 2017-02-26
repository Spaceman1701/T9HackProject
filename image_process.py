from PIL import Image
from PIL import ImageEnhance
import numpy as np
from PIL import ImageOps


def get_image_data(file_name):
    # Image trimming code found : http://stackoverflow.com/questions/9396312/use-python-pil-or-similar-to-shrink-whitespace
    im = Image.open(file_name)
    im = ImageOps.grayscale(im)
    im = ImageOps.autocontrast(im)

    pix = np.asarray(im)

    # pix = pix[:,:,0:3] # Drop the alpha channel
    idx = np.where(pix < 100)[0:2]  # Drop the color when finding edges

    box = list(map(min, idx))[::-1] + list(map(max, idx))[::-1]  # Bit fucked but that's okay
    # Normalize the box so that it produces a square image
    # Box = [left, upper, right, lower]
    l = box[0]
    u = box[1]
    r = box[2]
    low = box[3]
    upper_lower_dist = low - u
    left_right_dist = r - l

    # Generate the dimentions of a square image with size = largest between left-right and up-down size of cropped image.
    sq_box = [abs(l - int(upper_lower_dist / 2)), u, r + int(upper_lower_dist / 2),
              low]  # upper and lower have a greater distance
    if left_right_dist > upper_lower_dist:
        sq_box = [l, abs(u - int(left_right_dist / 2)), r, low + int(left_right_dist / 2)]

    region = im.crop(box)
    color = im.load()

    # Paste cropped image in square image so that the thumbnail will create a nice square image
    sq_img = Image.new('L', [sq_box[2] - sq_box[0], sq_box[3] - sq_box[1]], color=color[1, 1])

    # Need to put upper left corner to (new_size - old_size)/2
    if left_right_dist > upper_lower_dist:
        sq_img.paste(region, (0, int((left_right_dist - upper_lower_dist) / 2)))
    else:
        sq_img.paste(region, (int((upper_lower_dist - left_right_dist) / 2), 0))

    region_pix = np.asarray(sq_img)

    # To make a 20x20 sized image with the same ratio
    size = [20, 20]
    sq_img.thumbnail(size, Image.ANTIALIAS)

    new_size = [28, 28]
    new_img = Image.new("L", new_size, color=color[1, 1])
    new_img.paste(sq_img, (int((new_size[0] - size[0]) / 2),
                           int((new_size[1] - size[1]) / 2)))
    new_img = ImageOps.autocontrast(new_img)

    pix = new_img.load()

    pixels = []
    for i in range(0, 28):
        for j in range(0, 28):
            pixels.append(abs(pix[j, i] - 255) / 255)
    max_pixel = max(pixels)
    for p in range(0, len(pixels)):
        pixels[p] = (pixels[p] / max_pixel) - 0.1

    # Print to terminal
    for i in range(0, 28):
        print("")
        for j in range(28):
            if pixels[i * 28 + j] < 0.15:
                print("-", end="")
            else:
                print("@", end="")
    return pixels