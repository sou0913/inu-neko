# JPEG変換
from PIL import Image
import sys
import os
import re
import glob

input_path = "/Users/so/Desktop/projects/tf-app/kamo_materials/*/*.png"
output_path = "/Users/so/Desktop/projects/tf-app/kamo_materials/karugamo"
ls = glob.glob(input_path)
count = 1
for l in ls:
   img = Image.open(l)
   img.save(output_path + "/" + "converted" + str(count) + ".jpg", "jpeg")
   count += 1
   print(str(count) + "images are converted")
