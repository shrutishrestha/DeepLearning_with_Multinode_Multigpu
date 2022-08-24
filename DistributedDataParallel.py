import os
import csv
import cv2
import torch
from datetime import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cProfile, pstats
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

#for neural network
from torchvision import datasets, transforms

#mp
import hostlist
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def make(folderpathlist):
    for folderpath in folderpathlist:
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
            
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512,512)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(saturation=1.2, brightness=(0, 1), contrast=(0, 1)),
            transforms.GaussianBlur(kernel_size=(5,5)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)

        return image, label

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding=1, bias= False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()

        # enters from block 2
        if stride != 1 or in_planes != self.expansion * planes: #stride != 1, not first layer //// in_planes - 64, planes x self.expansion = 64 x 1 //64,64
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size = 1,stride = stride, bias= False), nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):

       # first layer
        h_x = self.conv1(x)
        h_x = self.bn1(h_x)
        h_x = self.relu(h_x)

       # second layer
        h_x = self.conv2(h_x)
        h_x = self.bn2(h_x)

       # adding identity / shortcut

        h_x += self.shortcut(x)
        y = self.relu(h_x)

        return y
    
class Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(Resnet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer1 = self.make_layer(block, 64, num_blocks[0], first_stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], first_stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], first_stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], first_stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def make_layer(self, block, planes, num_blocks, first_stride):
        other_strides = [1]
        strides = [first_stride] + other_strides * (num_blocks - 1)       # first stride is for size reduction, other strides are 1 incase of 18 and 34

        layers = [] 
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion       # for resnet 18 and 34, inchannel == outchannel, so expansion = 1
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        
        output = output.reshape(output.shape[0], -1)
        
        output = self.linear(output)
        
        return output
    
def ResNet34():
    return Resnet(BasicBlock, [3, 4, 6, 3], num_classes=1)

def average_gradients(model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
    
def main(args, rank, local_rank, size, hostnames, gpu_ids, NODE_ID, MASTER_ADDR):
        dist.init_process_group("gloo", init_method='env://', world_size=size, rank=rank)
        gpu = torch.device("cuda",local_rank)
        
        if rank == 0 and local_rank == 0:
            make([args["snapshot_directory"],args["snapshot_model_dir"], args["result_directory"], args["result_model_dir"]]) 

        # for dataset and dataloader
        traindataset = CustomImageDataset(transform = True, annotations_file = args["train_file"], img_dir = args["train_dir"])     
        valdataset = CustomImageDataset(annotations_file = args["val_file"], img_dir = args["val_dir"])     

        model = ResNet34().to(gpu)
        ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
        
        criterion = torch.nn.BCELoss()
        optimizer = optim.Adam(ddp_model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset,
                                                                        num_replicas= size,
                                                                        rank= rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valdataset,
                                                                        num_replicas= size,
                                                                        rank= rank)
        
        trainDataLoader = DataLoader(traindataset, batch_size=args["train_batch_size"], shuffle=False, num_workers=args["num_threads"], pin_memory=True, sampler=train_sampler)
        valDataLoader = DataLoader(valdataset, batch_size=args["val_batch_size"], shuffle=False, num_workers=args["num_threads"], pin_memory=True, sampler=val_sampler)
        
        best_ckpt = {"epoch":-1, "current_val_metric":0, "model":ddp_model.state_dict()}

        epoch = args["start_iters"]

        epoch_list = []
        datetime_list = []
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        lenTrainDataLoader = len(trainDataLoader)
        lenValDataLoader = len(valDataLoader)
        
        for epoch in range(args["max_epochs"]):
            print("epoch",epoch,  datetime.now())
            loss_sum = 0
            acc_sum = 0
            for i, batch in enumerate(trainDataLoader):
                image, label = batch
                label = label.to(torch.float)
                
                image = image.to(gpu, non_blocking=True)
                label = label.to(gpu, non_blocking=True)
        
                optimizer.zero_grad()
                logits = ddp_model(image).flatten()
                pred_probab = nn.Sigmoid()(logits)
                y_pred = (pred_probab>0.5)
                
                #forward pass
                loss = criterion(pred_probab, label)
                #backward and optimize
                loss.backward()
                average_gradients(ddp_model)
                optimizer.step()
                
                acc = accuracy_score(label.cpu(), y_pred.cpu())
                loss_sum += loss.item()
                acc_sum += acc.item()

            train_loss = round(loss_sum/lenTrainDataLoader,6)
            train_acc = round(acc_sum/lenTrainDataLoader,6)       
            
            ddp_model.eval()
            loss_sum = 0
            acc_sum = 0
            with torch.no_grad():
                for i, batch in enumerate(valDataLoader):
                    image, label = batch
                    label = label.to(torch.float)
                    
                    image = image.to(gpu, non_blocking=True)
                    label = label.to(gpu, non_blocking=True)
        
                    logits = ddp_model(image).flatten()
                    pred_probab = nn.Sigmoid()(logits)
                    y_pred = (pred_probab>0.5)

                    #forward pass
                    loss = criterion(pred_probab, label)

                    #backward and optimize
                    acc = accuracy_score(label.cpu(), y_pred.cpu())
                    loss_sum += loss
                    acc_sum += acc.item()


            val_loss = round(loss_sum/lenValDataLoader,6)
            val_acc = round(acc_sum/lenValDataLoader,6)  
            
            
            print(train_loss, val_loss, train_acc, val_acc)
            current_ckpt = {"epoch":epoch, "current_val_metric":val_acc, "model":ddp_model.state_dict()}

            if best_ckpt["current_val_metric"] < val_acc:
                best_ckpt = {"epoch":epoch, "current_val_metric":val_acc, "model":ddp_model.state_dict()}

            epoch_list.append(epoch)
            datetime_list.append(datetime.now())
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            scheduler.step(val_loss)
        print("all done", datetime.now())

        if args["sbatch"]:

            header = ["epoch", "epoch start datetime", "train_loss", "train_acc", "val_loss", "val_acc"]
            rows = zip(epoch_list, datetime_list, train_loss_list, train_acc_list, val_loss_list, val_acc_list)

            with open(os.path.join(args["result_model_dir"], "results.csv"), 'w') as csvfile: 
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(header) 
                for row in rows:
                    csvwriter.writerow(row) 
                    
            if  rank == 0:
                torch.save(current_ckpt, args["current_checkpoint_fpath"])
                torch.save(best_ckpt, args["best_checkpoint_fpath"])

args = {}
fake = False
cprof = True
args["sbatch"] = True
args["max_epochs"] = 4
args["device_id"] = 0
args["num_threads"] = 1

args["seed"] = 42
args["train_batch_size"] = 8
args["val_batch_size"] = 8
args["learning_rate"] = 0.01
args["start_iters"] = 0
args["set_affinity"] = True

if fake:
    args["train_file"]= "catdogs/fake/bc/train_all.csv"
    args["val_file"]= "catdogs/fake/bc/val_all.csv"
    args["train_dir"] = "catdogs/fake/bc/train_all"
    args["val_dir"] = "catdogs/fake/bc/val_all"
else:
    args["train_file"]= "catdogs/real/bc/train_all.csv"
    args["val_file"]= "catdogs/real/bc/val_all.csv"
    args["train_dir"] = "catdogs/real/bc/train_all"
    args["val_dir"] = "catdogs/real/bc/val_all"

job_id = os.getenv('SLURM_JOB_ID')
args["snapshot_directory"] = "/scratch/"+str(job_id)+"/snapshots/"
args["snapshot_model_dir"] = os.path.join(args["snapshot_directory"])
args["best_checkpoint_fpath"] = "/scratch/"+str(job_id)+"/snapshots/"+"/best_checkpoint.pth"
args["current_checkpoint_fpath"] = "/scratch/"+str(job_id)+"/snapshots/"+"/current_checkpoint.pth"
args["result_directory"] = "/scratch/"+str(job_id)+"/results/"
args["result_model_dir"] = os.path.join(args["result_directory"])
    
if __name__ == "__main__":
    # get SLURM variables
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    size = int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])

    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

    # get IDs of reserved GPU
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")

    # define MASTER_ADD & MASTER_PORT
    os.environ['MASTER_ADDR'] = hostnames[0]
    os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids))) # to avoid port conflict on the same node
        
    # get distributed configuration from Slurm environment
    NODE_ID = os.environ['SLURM_NODEID']
    MASTER_ADDR = os.environ['MASTER_ADDR']


    # display info
    if  rank == 0:
        print(">>> Training on ", len( hostnames), " nodes and ",  size, " processes, master node is ", MASTER_ADDR)
    print("- Process {} corresponds to GPU {} of node {}".format( rank,  local_rank, NODE_ID))
    print("rank",rank)
    print("local_rank",local_rank)
    print("size",size)

    print("hostnames",hostnames)
    print("gpu_ids",gpu_ids)
    
        
    if cprof:
        profiler = cProfile.Profile()
        profiler.enable()
        main(args, rank, local_rank, size, hostnames, gpu_ids, NODE_ID, MASTER_ADDR)
        profiler.disable()
        stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
        stats.print_stats(40)
    else:
        main(args, rank, local_rank, size, hostnames, gpu_ids, NODE_ID, MASTER_ADDR)
