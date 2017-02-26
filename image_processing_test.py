# Author: Maxine Hartnett

from PIL import Image
from PIL import ImageEnhance
import numpy as np
from PIL import ImageOps


#Image trimming code found : http://stackoverflow.com/questions/9396312/use-python-pil-or-similar-to-shrink-whitespace
im = Image.open("berk_9.jpg")
im = ImageOps.grayscale(im)
#im = ImageOps.autocontrast(im)

pix = np.asarray(im)


#pix = pix[:,:,0:3] # Drop the alpha channel
idx = np.where(pix<100)[0:2] # Drop the color when finding edges

box = list(map(min,idx))[::-1] + list(map(max,idx))[::-1] # Bit fucked but that's okay
# Normalize the box so that it produces a square image
# Box = [left, upper, right, lower]
upper_lower_dist = box[3] - box[1]
left_right_dist = box[2] - box[0]

sq_box = [abs(box[0]-int(upper_lower_dist/2)), box[1], box[2]+int(upper_lower_dist/2), box[3]] # upper and lower have a greater distance
if left_right_dist > upper_lower_dist:
    print("left right bigger")
    sq_box = [box[0], abs(box[1]-int(left_right_dist/2)), box[2], box[3]+int(left_right_dist/2)]

region = im.crop(sq_box)
region_pix = np.asarray(region)

# To make a 20x20 sized image with the same ratio
size = [20, 20]
region.thumbnail(size, Image.ANTIALIAS)

new_size = [28, 28]
new_img = Image.new("L", new_size, color=225)
new_img.paste(region, (int((new_size[0]-size[0])/2),
                     int((new_size[1]-size[1])/2)))
new_img.show()
new_img = ImageOps.autocontrast(new_img)
new_img.show()
pix = new_img.load()
pixels = []
for i in range(0, 28):
    for j in range(0, 28):
        pixels.append(abs(pix[j, i]-255)/265)
max_pixel = max(pixels)
for p in range(0, len(pixels)):
    pixels[p] = pixels[p]/max_pixel

for i in range(0, 28):
    print("")
    for j in range(28):
        if pixels[i * 28 + j] < 0.3:
            print("-", end="")
        else:
            print("@", end="")
print(len(pixels))

