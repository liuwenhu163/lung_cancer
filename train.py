import argparse as args
from data import MyLidcDataset
import torch
from torch.utils.tensorboard import SummaryWriter
from data import MyLidcDataset
import argparse
from model import model as model
from fcn import FCN32s as fcn
from r2u_net import model as r2u_net
from u2a_net import model as u2a_net
from u_net import model as u_net
from nnu_net import model as nnu_net
from nestedunet import model as nestedunet
from att_unet import model as att_unet
import torch.backends.cudnn as cudnn
#from model import NestedUNet
# from model import ConvNeXt
"""todo index of the paper and loss function"""
from paper2picture import BCEDiceLoss,iou_score,dice_coef,dice_coef2
import monai
import tqdm
import random
#from transunet import VisionTransformer as ViT_seg
#from transunet import CONFIGS as CONFIGS_ViT_seg

device = "cuda"
def train_test(model,train_dataloader,test_dataloader,loss,optimizer,writer,model_name):

    best_loss = 10000


    for i in range(150):
        train_loss,test_loss,train_iou,test_iou,train_dice,test_dice = 0.0,0.0,0.0,0.0,0.0,0.0
        model.train()
        tq = tqdm.tqdm(total=len(train_dataloader))
        tq.set_description(f'Train:epoch{i:4},LR:{optimizer.param_groups[0]["lr"]:0.6f}')
        for batch_idx,(data,target) in enumerate(train_dataloader):
            data,target = data.to(device),target.to(device)

            if len(data[0]) == 0:
                continue
            optimizer.zero_grad()
            prediction = model(data)


            import pdb
            # pdb.set_trace()
            losses = loss(prediction,target)

            iou = iou_score(prediction,target)
            dice = dice_coef(prediction,target)
            
            losses.backward()
            optimizer.step()
            train_loss += losses
            train_iou += iou
            train_dice += dice
            tq.update(1)
            tq.set_postfix(train_loss=f'{train_loss/(batch_idx+1):4.6f}',
                            train_iou=f'{train_iou/(batch_idx+1):4f}',
                            train_dice=f'{train_dice/(batch_idx+1):4.4f}')
        tq.close()
        writer.add_scalar("Loss/train",train_loss/len(train_dataloader),i)
        writer.add_scalar("IOU/train",train_iou/len(train_dataloader),i)
        writer.add_scalar("dice/train",train_dice/len(train_dataloader),i)

        with torch.no_grad():
            model.eval()
            
            tq = tqdm.tqdm(total=len(test_dataloader))
            for batch_idx, (data, target) in tqdm.tqdm(enumerate(test_dataloader)):
                data, target = data.to(device), target.to(device)
                if len(data[0]) == 0:
                    continue
                prediction = model(data)


                losses = loss(prediction,target)
                iou = iou_score(prediction,target)
                dice = dice_coef(prediction,target)
                test_loss += losses
                test_iou += iou
                test_dice += dice
                tq.update(1)
                tq.set_postfix(test_loss=f'{test_loss/(batch_idx+1):4.6f}',
                              test_iou=f'{test_iou/(batch_idx+1):4f}',
                              test_dice=f'{test_dice/(batch_idx+1):4.2f}')
            test_loss = test_loss/len(test_dataloader)
            
            if test_loss<best_loss:
                best_loss = test_loss
                torch.save(model, model_name + ".hdf5")
            tq.close()
            writer.add_scalar("Loss/test",test_loss,i)
            writer.add_scalar("IOU/test",test_iou/len(test_dataloader),i)
            writer.add_scalar("dice/test",test_dice/len(test_dataloader),i)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cancer predict")
    parser.add_argument('--path',default="/root/autodl-tmp/new_data/",type=str,help= "train file path")
    
    parser.add_argument('--model',default='model',type=str,help='choose model')
    parser.add_argument('--batch_size', default=1,type=int, help="batch size")
    parser.add_argument('--number_workers', default=16,type=int, help="number of workers")
    parser.add_argument('--lr', default=3e-4, type=float, help="learning rate")
    parser.add_argument('--input_channels', default=1, type=int, help="input channels")
    parser.add_argument('--hidden_channels', default=32, type=int, help="hidden channels")
    parser.add_argument('--output_channles', default=1, type=int, help="output channels")
    parser.add_argument('--image_shape', default=[512,512], type=int, help="output channels")
    torch.manual_seed(42)
    
    torch.cuda.manual_seed(42)
    args = parser.parse_args()
    train_path = args.path + "training"
    test_path = args.path + "testing"
    batch_size = args.batch_size
    lr = args.lr
    writer = SummaryWriter(comment='_' + args.model)
    number_workers = args.number_workers
    random.seed(42)
    image_shape = args.image_shape
    train_data = MyLidcDataset(train_path,Albumentation=False)
    test_data = MyLidcDataset(test_path,Albumentation=False)
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=4,num_workers=16,shuffle=True,drop_last=True,
                                                   pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_data,batch_size=4,num_workers=16,shuffle=False,drop_last=False,
                                                  pin_memory=True)
    train_model = model(1)
    if args.model == "model":
       train_model = model(1)
    if args.model == "unet":
		train_model = unet(1)
	if args.model == "r2u_net":
		train_model = r2u_net(1)
	if args.model == "nestedunet":
		train_model = nestedunet(1)
	if args.model == "u2a_net":
		train_model = u2a_net(1)
	if args.model == "nnu_net":
		train_model = nnu_net(1)
	if args.model == "fcn":
		train_model = fcn(1)
	if args.model == "att_unet":
		train_model = att_unet(1)
    train_model = train_model.to(device)
    loss = BCEDiceLoss().cuda()
    cudnn.benchmark = True
    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

    train_test(train_model,train_dataloader,test_dataloader,loss,optimizer,writer,args.model)


