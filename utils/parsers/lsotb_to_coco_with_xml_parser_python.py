import pandas as pd
import os
import pandas_read_xml as pdx
import json
import string
import itertools
import glob
from glob import glob
import PIL.Image as Image
import numpy as np
import cv2

print(os.getcwd())


MAX_N = 1

categories = [
    {
        "supercategory": "none",
        "name": "lsotb",
        "id": 0
    }
]

#paths
root_path_img ="LSOTB_TIR_train/images/" +"part_5" 
root_path_ann = os.path.join("LSOTB_TIR__annotations/", "part_5")

#path_videos = os.path.join(os.getcwd(), "LSOTB_video_processing")
#os.mkdir(path_videos)


json_file = "train_LSOTB.json"



res_file  = {
    "categories": [],
    "images": [],
    "annotations": []
    }
                             
                             
                             
annot_count = 0
image_id = 0
processed = 0   



for  division in os.listdir(root_path_ann):    

    if len(os.listdir(os.path.join(root_path_ann, division))) == 0:

       #print("Directory is empty")

       print ( f'{os.path.join(root_path_ann, division)}  is  empty' )

    else:    

       
       for folder in os.listdir( os.path.join(root_path_ann, division)):  

           if len(os.listdir(os.path.join(root_path_ann, division, folder))) == 0:
              
              print ( f'{os.path.join(root_path_ann, division, folder)}  is  empty' )
           
           else:
              
              
              print(folder)
              img_array=[]
              #os.mkdir(nouveau_dossier)

              for count, filename_ann in enumerate(glob(f'{os.path.join(root_path_ann, division,folder)}/*.xml')):
              #for filename in enumerate(os.path.join(root_path_ann, division,folder, ''*.xml')): 
                  print(filename_ann)
                  #rename filename_ann
            
                  gt_path_file = pdx.read_xml(filename_ann) 
                  gt_path_file = gt_path_file.to_json()
                  gt_path_file = json.loads(gt_path_file)
                  print(json.dumps(gt_path_file, indent=4))
            
                  base_path = gt_path_file["annotation"]["0"]["folder"]

                  #base_path = os.path.join(division, folder)
            
                             
                  file_path = gt_path_file["annotation"]["0"]["filename"]  
            
            
                  #img_path = os.path.join(root_path_img, base_path, file_path + ".jpg")
            
                  #'''
            
                  #rename  files
            
                  #rename image
             
                  img_path1 = os.path.join(root_path_img, base_path, file_path + ".jpg")
            
                  dst_part1 = f"{root_path_img.split('/')[-1]}_train_{base_path.split('/')[0][-1]}_{folder}_"
            
                  dst_part2 = file_path + ".jpg"
            
                  dst= f"{dst_part1}{dst_part2}"

                  image_renamed = dst
            
                  print(image_renamed)
            
                  #dst=f"{root_path_img}/{division}/{folder }/{dst}"

                  #dst=f"{root_path_img}/{division}/{folder }/{image_renamed}"
            
                  image_renamed_path =  f"{root_path_img}/{division}/{folder }/{image_renamed}"
                  #image_renamed_path =  f"{root_path_img}/{division}/{folder }/{nouveau dossier}/{image_renamed}"
                  
                  #dst=f"{root_path_img}/{base_path }/{dst}"
                 
                  os.rename(os.path.join(os.getcwd(), img_path1), os.path.join(os.getcwd(), image_renamed_path) )
                 # os.rename(img_path1, image_renamed_path )

                  
            
                  #file_renamed_path = dst                 
            
                  #filename_image = os.path.join("part_5", base_path, image_renamed) 
            
            
                  #''
          
           
                  #path after renaming   
            
            
                  img_path = glob(os.path.join(root_path_img, division,folder)+ '/*.jpg')[count]

                  print(img_path)

            
                  filename_image = f"{img_path}".split('/')[2]
            
                  print(filename_image)
            
            
                  img_elem = {"file_name": filename_image,
                       "height": gt_path_file["annotation"]["0"]["size"]["height"],
                       "width": gt_path_file["annotation"]["0"]["size"]["width"],
                       "id": image_id}                  
                             
                  print(img_elem)  
            
                  res_file["images"].append(img_elem)


                  xmin = int(gt_path_file["annotation"]["0"]["object"]["bndbox"]["xmin"])
                  ymin = int(gt_path_file["annotation"]["0"]["object"]["bndbox"]["ymin"])
                  xmax = int(gt_path_file["annotation"]["0"]["object"]["bndbox"]["xmax"])
                  ymax = int(gt_path_file["annotation"]["0"]["object"]["bndbox"]["ymax"])

                  w = xmax - xmin
                  h = ymax - ymin
                  area = w * h
                  poly = [[xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax]]
            

                  annot_elem = {
                    "id": annot_count,
                    "bbox": [
                        float(xmin),
                        float(ymin),
                        float(w),
                        float(h)
                    ],
                    "segmentation": list([poly]),
                    "image_id": image_id,
                    "ignore": 0,
                    "category_id": 0,
                    "iscrowd": 0,
                    "area": float(area),
                    "object_type" : gt_path_file["annotation"]["0"]["object"]["name"],
                    "istrackid": gt_path_file["annotation"]["0"]["object"]["trackid"],
                    "isoccluded": gt_path_file["annotation"]["0"]["object"]["occluded"]

                    }

                  res_file["annotations"].append(annot_elem)

                  annot_count += 1

                  image_id += 1

                  processed += 1

        

        
                  #make a video from frame

                  img = Image.open(img_path)
                  #img_w, img_h = img.size
                  (width, height) = img.size
                  size = (width, height)
                  imgcv = cv2.imread(img_path)
                  #img.shape

                  img_array.append(imgcv)



              out=cv2.VideoWriter('LSOTB_video_processing/' + f"{root_path_img.split('/')[-1]}_train_{base_path.split('/')[0][-1]}_{folder}"+ '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size )
              for i in range(len(img_array)):
                  out.write(img_array[i])
              out.release()    

            
 

    

with open(json_file, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)   


print("Processed {} {} images...".format(processed, "train"))
print("Done.")

                        
            




