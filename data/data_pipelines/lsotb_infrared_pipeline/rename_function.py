import os
import string


def image_rename(gt_path_file, root_path):
    """rename an image
    gt_path_file :JSON object
    root_path :
    # root_path ="LSOTB_TIR_train/images/" +"part_5"
    #root_path =os.getcwd()
    #os.path.join(os.getcwd(),root_path)
    """
    #Directory=os.getcwd()

    
    base_path= gt_path_file["annotation"]["0"]["folder"]

    folder = base_path.split('/')[-1]

    #filename = f'{folder}/*.jpg'
    filename = f'{folder}/00000001.jpg'
    

    dst_part1 = f"{root_path.split('/')[-1]}_train_{base_path.split('/')[0][-1]}_{folder}_"
            
    dst_part2= f"{filename}".split('/')[-1]

    dst= f"{dst_part1}{dst_part2}"

    dst=f"{root_path}/{folder }/{dst}"
            #.split('\\')[-1].split('.')[-2]
            #dst = f"part_{5}_train_{1}_{folder}_{str(count)}.jpg"
            #src = f"{Directory}/{folder }/{filename}/"
            #dst= glob(f"{Directory}/{folder }/{dst}")
    os.rename( filename, dst )
    file_renamed_path = dst

    return file_renamed_path 
