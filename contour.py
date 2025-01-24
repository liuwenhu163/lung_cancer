import cv2
import glob 
import os
import torch
file_path = glob.glob(os.path.join("/root/autodl-tmp/crop","*","masks","*"))
for i in file_path:
    image = cv2.imread(i)
    image = torch.from_numpy(image[:,:,0]/255)
    if torch.sum(image)<160:
        os.remove(i)
        os.remove(i.replace("masks","images"))
	
    
    
    
