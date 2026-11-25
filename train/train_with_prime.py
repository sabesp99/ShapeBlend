import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import argparse
from torchvision.models import resnet50
from torchvision.models import convnext_tiny
import os
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import InterpolationMode
import sys
import torch.distributed as dist
import random
import augmentations
import numpy as np
import torch.nn.functional as F
from PIL import Image
from rand_filter import RandomFilter
from color_jitter import RandomSmoothColor
from diffeomorphism import Diffeo

from prime import GeneralizedPRIMEModule
from prime import PRIMEAugModule
from prime import TransformLayer
from tqdm import tqdm
corruptions = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
                    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
                    
severities =["1", "2", "3", "4", "5"]

parser = argparse.ArgumentParser()
parser.add_argument('--exp', dest='exp', type=str, help="directory with 'train' 'val' 'valC' dataset", default = "")
parser.add_argument('--num', dest='num', type=str, help="directory with 'train' 'val' 'valC' dataset", default = "")
parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='only evaluation')
parser.add_argument('--corr', dest='corr', default=False, action='store_true', help='evaluates on corrupted dataset, true only if eval true')
parser.add_argument('--nclass', dest='nclass', type=int, help='numclasses', default = 200)
parser.add_argument('--cpus', dest='cpus', type=int, help='n cpu', default = 8)
parser.add_argument('--sched', dest='sched', type=str, help='evaluates on corrupted dataset, true only if eval true', default = "step")

parser.add_argument('--epochs', dest='epochs', type=int, help='batch size', default = 20)
parser.add_argument('--batch', dest='batch', type=int, help='batch size', default = 64)
#--------------------------SGD PARAMETERS-----------------------------------------
parser.add_argument('--lr', dest='lr', type=float, help='learning rate', default = 0.1)
parser.add_argument('--m', dest='momentum', type=float, help='momentum', default = 0.9)
parser.add_argument('--wd', dest='wdecay', type=float, help='weight decay', default = 1e-5)
#--------------------------STEP SCHEDULER PARAMETERS---------------------------------
parser.add_argument('--step', dest='lrstep', type=float, help='learning rate step size', default = 30)
parser.add_argument('--gamma', dest='lrgamma', type=float, help='learning rate gamma', default = 0.1)
#-------------------------PARALLELISM PARAMETERS-------------------------------------
parser.add_argument('--num-nodes', type=int, default=1,help='Number of available nodes/hosts')
parser.add_argument('--node-id', type=int, default=0,help='Unique ID to identify the current node/host')
parser.add_argument('--num-gpus', type=int, default=4,help='Number of GPUs in each node')
#-------------------------METHODOLOGY PARAMETERS-------------------------------------
parser.add_argument('--deepaugment', dest='deepaugment', default=False, action='store_true', help='only evaluation')
parser.add_argument('--augmix', dest='augmix', default=False, action='store_true', help='only evaluation')
parser.add_argument('--online', dest='online', default=False, action='store_true', help='only evaluation')
parser.add_argument('--prime', dest='prime', default=False, action='store_true', help='only evaluation')
#-------------------------AUGMIX PARAMETERS-------------------------------------
parser.add_argument('--allops', dest='allops', default=False, action='store_true', help='only evaluation')
parser.add_argument('--mw', default=3, type=int, help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--md', default=-1, type=int, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--augs', default=1, type=int, help='Severity of base augmentation operators')
parser.add_argument('--augpc', default=1., type=float,help='Probability distribution coefficients')

args = parser.parse_args()
WORLD_SIZE = args.num_gpus * args.num_nodes
os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '9956' 
args.batch = int(args.batch / args.num_gpus)

def aug(image, preprocess):
    aug_list = augmentations.augmentations
    if args.allops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(
    np.random.dirichlet([args.augpc] * args.mw))
    m = np.float32(np.random.beta(args.augpc, args.augpc))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mw):
        image_aug = image.copy()
        depth = args.md if args.md > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, args.augs)
    # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""
  def __init__(self, dataset, preprocess):
    self.dataset = dataset
    self.preprocess = preprocess

  def __getitem__(self, i):
    x, y = self.dataset[i]

    im_tuple = (self.preprocess(x), aug(x, self.preprocess), aug(x, self.preprocess))
    return im_tuple, y

  def __len__(self):
    return len(self.dataset)



class OnlineDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform Online augmentation."""
    def __init__(self, root, transform=None):
        self.levels = ["blend1", "blend2", "blend3", "blend4", "blend5", "blend6", "blend7", "blend8", "blend9", "negative1", "negative2", "negative3", "negative4", "negative5", "negative6", "negative7", "negative8", "negative9"] # Percorsi delle immagini "clean" e "blend"
        self.root = root
        self.transform = transform
        self.imgs = []
        self.labels = []
        base_dir = os.path.join(root, self.levels[0], "train")
        self.classes = sorted(entry.name for entry in os.scandir(base_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_path = os.path.join(base_dir, cls_name)
            for fname in os.listdir(cls_path):
                path = os.path.join(cls_name, fname)
                self.imgs.append(path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
            img_path = self.imgs[idx]
            label = self.labels[idx]
            level = random.choice(self.levels)
            path = os.path.join(self.root, level, "train", img_path)
            image = Image.open(path).convert("RGB") 
            if self.transform:
                image = self.transform(image)

            return image, label


#----------------------------------------------------------------------
def printf(line, rank):
    if rank==0:
        with open(f"train{args.nclass}/{args.exp}_{args.num}.txt", 'a') as file:
            file.write(line)
# Train
def train_loop(dataloader, model, loss_fn, optimizer, scheduler, device,gr,prime):
    model.train()
    for X, y in tqdm(dataloader, desc=f"Training epoch"):
        if not args.augmix:
            X, y = X.to(device), y.to(device)
            pred = model(prime(X))
            loss = loss_fn(pred,y)
        else:
            X_all = torch.cat(X,0).to(device)
            y = y.to(device)
            pred_all = model(X_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
            pred_all, X[0].size(0))
            loss = loss_fn(logits_clean,y)

            p_clean = F.softmax(logits_clean,dim=1)
            p_aug1 = F.softmax(logits_aug1,dim=1)
            p_aug2 = F.softmax(logits_aug2,dim=1)
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
# Validation
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = int(len(dataloader.dataset)/args.num_gpus)
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc=f"Validation"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y)
            correct += (pred.argmax(1) == y).type(torch.float).sum()
    test_loss /= num_batches
    correct /= size
    accuracy = 100*correct
    return accuracy, test_loss
#--------------------------------------------------------------------------------

base_dir = f"./../imagenet{args.nclass}"
clean_dir = base_dir + f"/clean/train"
EDSR_clean_dir = base_dir + f"/EDSR/clean/train"
CAE_clean_dir = base_dir + f"/CAE/clean/train"
EDSR_online_dir = base_dir + f"/EDSR/"
CAE_online_dir = base_dir + f"/CAE/"
val_dir = base_dir  + f"/clean/val" #if not args.online else base_dir + f"/clean/3xval"

if os.path.exists(f"weights{args.nclass}/last{args.num}_{args.exp}.pth"):
    check = True
else:
    check = False

# Creates dataset and dataloader
INmean=[0.485, 0.456, 0.406]
INstd=[0.229, 0.224, 0.225]

if args.augmix or args.prime:
    train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(0.5),
    ])

else:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(0.5),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean = INmean, std=INstd),])     
    
preprocess = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean = INmean, std=INstd)])

val_transform = transforms.Compose([
   transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
   transforms.CenterCrop(224),
   transforms.PILToTensor(),
   transforms.ConvertImageDtype(torch.float),
   transforms.Normalize(mean = INmean, std=INstd),])

def worker(local_rank, args):
    global_rank = args.node_id * args.num_gpus + local_rank 
    dist.init_process_group(backend='nccl',  world_size=WORLD_SIZE, rank=global_rank) 
    device = torch.device("cuda:" + str(local_rank))

    nwt=int(args.cpus*3/4)
    nwv = int(args.cpus/4)

    train_dataset = torchvision.datasets.ImageFolder(root=clean_dir, transform=train_transform)
    concat = []
    if args.deepaugment:
        EDSR_clean_dataset = torchvision.datasets.ImageFolder(root=EDSR_clean_dir, transform=train_transform)
        CAE_clean_dataset = torchvision.datasets.ImageFolder(root=CAE_clean_dir, transform=train_transform)
        concat.append(EDSR_clean_dataset)
        concat.append(CAE_clean_dataset)
        if args.online:
            EDSR_online_dataset = OnlineDataset(root=EDSR_online_dir, transform=train_transform)
            CAE_online_dataset = OnlineDataset(root=CAE_online_dir, transform=train_transform)
            concat.append(EDSR_online_dataset)
            concat.append(CAE_online_dataset)
    if args.online:
        online_dataset = OnlineDataset(root=base_dir, transform=train_transform)
        concat.append(online_dataset)
    if args.online or args.deepaugment:
        concat.append(train_dataset)
        train_dataset = torch.utils.data.ConcatDataset(concat)
    if args.augmix:
        train_dataset = AugMixDataset(train_dataset, preprocess)
    if args.prime:
        augmentat = []
        diffeo = Diffeo(sT=1, rT=1, scut=1, rcut=1, cutmin=2, cutmax=500, alpha=1.0, stochastic=True)
        augmentat.append(diffeo)
        color = RandomSmoothColor(cut=500, T=0.05, freq_bandwidth=20, stochastic=True)
        augmentat.append(color)
        filt = RandomFilter(kernel_size=3,sigma=4.0, stochastic=True)
        augmentat.append(filt)
        prime_module = GeneralizedPRIMEModule(
            preprocess=TransformLayer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            mixture_width=args.mw,
            mixture_depth=args.md, max_depth=3,
            aug_module=PRIMEAugModule(augmentat),
        )
        prime_module.to(device)
        
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=WORLD_SIZE,rank=global_rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=int(nwt/args.num_gpus), pin_memory=True, drop_last=False, sampler=train_sampler)

    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=val_transform)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,num_replicas=WORLD_SIZE,rank=global_rank)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, num_workers=int(nwv/args.num_gpus), pin_memory=True, drop_last=False, sampler=val_sampler)
    #----------------------------------------------------------------
    
    # Defines the model, loss functions optimizator and scheduler
    #model = resnet50(weights = 'IMAGENET1K_V1')
    model = convnext_tiny(weights = None)
    state_dict = torch.load("weights1000/tiny.pth")
    model.load_state_dict(state_dict)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.wdecay, nesterov=True)
    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader), 1, 1e-6 / (args.lr * args.batch / 256.)))
    

    acc, best_acc , no_imp, did= 0, 0, 0, 0

    if check:
        checkpoint = torch.load(f"weights{args.nclass}/last{args.num}_{args.exp}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        did = checkpoint['did']
        best_acc = checkpoint['best_acc']
        no_imp = checkpoint['no_imp']
        printf(f"\nRestarting from checkpoint.\n", global_rank)

    

    else:
        printf(f"N samples training set: {len(train_loader.dataset)}, num workers: {nwt}, GPUs: {args.num_gpus}\n"+
            f"N samples validation set: {len(val_loader.dataset)}, num workers: {nwv}\n" +
            f"Online Augmentation: {args.online}\tAugmix Augmentation: {args.augmix}\Prime Augmentation: {args.prime}\tDeepAugment Augmentation: {args.deepaugment}\n"+
            f"Optimizer: SGD\tlearning rate: {args.lr}\tmomentum: {args.momentum}\tweight decay: {args.wdecay}\n"+
            f"Scheduler: {args.sched}\tgamma: {args.lrgamma}\tstep size: {args.lrstep}\n"+
            f"Epochs: {args.epochs}\t\texp: {args.exp}\t\tnum: {args.num}\t\tbatch: {args.batch}\t\tclasses: {args.nclass}\n", global_rank)

    scheduler.step(did * len(train_loader))
    #------------------------TRAIN----------------------------------
    for t in range(did,args.epochs):
        train_loader.sampler.set_epoch(t)
        printf(f"\nEpoch {t+1}/{args.epochs}\tlr: {scheduler.get_last_lr()}\n", global_rank)
        train_loop(train_loader, model, loss_fn, optimizer, scheduler, device, global_rank, prime_module)
        acc, loss = test_loop(val_loader, model, loss_fn, device)
        torch.distributed.all_reduce(acc, op=dist.ReduceOp.AVG)
        torch.distributed.all_reduce(loss, op=dist.ReduceOp.AVG)
        printf(f"Validation Accuracy: {(acc):>0.3f}%, Avg loss: {loss:>8f}\nno improvements for {no_imp} epochs\n", global_rank)

        if acc > best_acc:
            best_acc = acc.item()
            if global_rank==0:
                torch.save(model.state_dict(), f"weights{args.nclass}/best{args.num}_{args.exp}.pth")
            no_imp = 0
        
        no_imp+=1

        if global_rank==0:
            torch.save({
                    'did': t+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc' : best_acc,
                    'no_imp' : no_imp,
                }, f"weights{args.nclass}/last{args.num}_{args.exp}.pth")

    printf("Done!\n", global_rank)
        

if __name__ == '__main__':
    torch.multiprocessing.spawn(worker, nprocs=args.num_gpus, args=(args,))