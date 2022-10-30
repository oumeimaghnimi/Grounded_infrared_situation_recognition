import cv2
import numpy as np
import glob
import PIL.Image as Image
from os import listdir

G_img=listdir('/content/gdrive/MyDrive/Colab Notebooks/first trial/basics_for_infrared_detection/dataset/LSOTB-TIR_TrainingData_part5/TIR_training_004_extracted/TIR_training_004') 

for item  in G_img:
    target_dir = os.path.join(repo_root_data_path_part_5, item);
    img_array=[]
    for filename in glob.glob(f'{target_dir}/*.jpg'):
        img = Image.open(filename)
        (width, height)=img.size
        size=(width, height)
        imgcv=cv2.imread(filename)
        #img.shape
        img_array.append(imgcv)
    out=cv2.VideoWriter(item + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release() 
