import torch
from model import model
import cv2 as cv
from data import MyLidcDataset
import tqdm
import argparse as args
import os

parser = argparse.ArgumentParser(description="cancer predict")
parser.add_argument('--path',default="/root/autodl-tmp/new_data/",type=str,help= "train file path")    
parser.add_argument('--model',default='model',type=str,help='choose model')
parser.add_argument('--save_path',default="/root/autodl-tmp/results",type=str,help = "save result path")
val_path = args.path + "validation"
save_path = args.save_path
if not os.isdir(save_path):
	os.makedirs(savepath)
val_dataloader = MyLidcDataset(val_path)
val_dataloader = torch.utils.data.DataLoader(val_dataloader,batch_size=1,num_workers=16,shuffle=False,drop_last=False,
                                                   pin_memory=True)
sigmoid =  torch.nn.Sigmoid()
# model = model(1).to("cuda")
model = torch.load("best.hdf5")
# model = torch.load(args.model+".hdf5")
with torch.no_grad():
    model.eval()
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_dataloader)):
        data, target = data.to("cuda"), target.to("cuda")
        if len(data[0]) == 0:
           continue
        prediction = model(data)
	    prediction = sigmoid(prediction)*180
		prediction = torch.cat((prediction,torch.zeros(1,2,224,224).to("cuda")),dim=1)
        prediction = data*255 + prediction
        
        prediction = prediction[0].permute(1,2,0)
        cv.imwrite(save_path+"/{}.png".format(batch_idx),prediction.cpu().numpy())

        