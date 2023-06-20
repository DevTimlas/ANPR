import os
import easyocr
import numpy as np
import cv2
import torch
import shutil
import sys

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best2.pt', force_reload=True)

# img = '7_0.jpg'
img = str(sys.argv[1])

results = model(img)

# result.save()
# results.crop(save=True, save_dir='crop')


save_pth = os.path.join(os.getcwd(), 'crop')
crop_dir = (os.path.join(save_pth, 'crops', 'licence_plate/'))
if os.path.exists(save_pth):
	shutil.rmtree(save_pth)	

sav = results.crop(save=True, save_dir=save_pth)

def filter_txt(region, ocr_result, region_threshold):
    rect_size = region.shape[0]*region.shape[1]

    plate = []

    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length*height / rect_size > region_threshold:
            plate.append(result[1])


    return plate
    

reader = easyocr.Reader(['en'], gpu=False)

if os.path.exists(crop_dir):
    for fname in os.listdir(crop_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            full = (crop_dir + fname)
            rg = cv2.imread(full)

            result = reader.readtext(rg)

            txt = (filter_txt(rg, result, 0.5))
            for i in txt:
                print(''.join(i))
