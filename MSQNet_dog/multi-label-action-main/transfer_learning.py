import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
# import clip
import pandas as pd
import math
import time
import copy

from torch.utils.data import DataLoader
#from custom_dataset import CustomDataset, Dog
from datasets.datasets import Dog
from datasets.datamanager import DataManager
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from torchvision.models import VisionTransformer
from torchmetrics.classification import MultilabelAveragePrecision
from datasets.transforms_ss import *
from torchvision.transforms import Compose
from transformers import CLIPProcessor, CLIPModel, TimesformerModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel, logging
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.utils import AverageMeter
from datasets.datamanager import DataManager


class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

def _get_prompt(cl_names):
    temp_prompt = []
    for c in cl_names:
        prompt = "a dog is " + c
        print("prompt",prompt)
        temp_prompt.append(prompt)
    return temp_prompt
    
def _get_text_features(cl_names):
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    act_prompt = _get_prompt(cl_names)
    texts = tokenizer(act_prompt, padding=True, return_tensors="pt")
    text_class = text_model(**texts).pooler_output.detach()
    return text_class

def append_files(names, file_path, csv):
    names_list = []
    for name in names:
        if csv:
            path = os.path.join(file_path, f"annotations-{name}.csv")
            file = pd.read_csv(path, header=None, skiprows=1)
        else:
            file = os.path.join(file_path, name)
        names_list.append(file)
    return names_list

def get_train_transforms():
        """Returns the training torchvision transformations for each dataset/method.
           If a new method or dataset is added, this file should by modified
           accordingly.
        Args:
          method: The name of the method.
        Returns:
          train_transform: An object of type torchvision.transforms.
        """
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        input_size = 224
        unique = Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                            GroupRandomHorizontalFlip(True),
                            GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                            GroupRandomGrayscale(p=0.2),
                            GroupGaussianBlur(p=0.0),
                            GroupSolarization(p=0.0)])
        common = Compose([Stack(roll=False),
                        ToTorchFormatTensor(div=True),
                        GroupNormalize(input_mean, input_std)])
        transforms = Compose([unique, common])
        return transforms
    
def get_test_transforms():
    """Returns the evaluation torchvision transformations for each dataset/method.
        If a new method or dataset is added, this file should by modified
        accordingly.
    Args:
        method: The name of the method.
    Returns:
        test_transform: An object of type torchvision.transforms.
    """
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    input_size = 224
    scale_size = 256

    unique = Compose([GroupScale(scale_size),
                        GroupCenterCrop(input_size)])
    common = Compose([Stack(roll=False),
                        ToTorchFormatTensor(div=True),
                        GroupNormalize(input_mean, input_std)])
    transforms = Compose([unique, common])

    return transforms

def save(model, epoch):
    # backbone_state_dict = self.model.backbone.state_dict()
    # linear_state_dict = self.model.linear1.state_dict()
    # transformer_state_dict = self.model.transformer.state_dict()
    # query_embed_state_dict = self.model.query_embed.state_dict()
    # group_linear_state_dict = self.model.fc.state_dict()
    # optimizer_state_dict = self.optimizer.state_dict()
    torch.save(model.state_dict(), f"./checkpoints/state_dict_epoch{epoch}")
    torch.save(model, f"./checkpoints/model_epoch{epoch}")

def save_to_csv(predictions, path):
     df = pd.DataFrame(predictions, columns=['Dog_id', 'Start frame', 'End frame', 'Prediction', 'Ground truth'])
     sorted_on_id = df.groupby('Dog_id')
     print("preds", predictions)
     print("sorted", sorted_on_id)

     for dog_path, dog_df in sorted_on_id:
          dog_split = dog_path.split('/')
          dog_name = dog_split[-1]
          file_name = f"predictions-{dog_name}.csv"
          output = os.path.join(path, file_name)
          dog_df.to_csv(output, index=False)
          print(f"saved to {output}")

def test(model, test_loader, device):
    model.eval()
    eval_meter = AverageMeter()
    eval_metric = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
    eval_metric.to(device)
    predictions = [] # col 1 has dog id, col 2 start frame, col 3 end frame, col 4 predicted label, col 4 ground truth
    for data, label, info in test_loader:
        data, label = data.to(device), label.long().to(device)
        with torch.no_grad():
            output = model(data)
        # print("output", output)
        # print("label test og", label)
        highest_output = torch.argmax(output, dim=1) # for single label classification
        og_label_output = torch.argmax(label, dim=1)

        for idx in range(label.shape[0]):
            dog_id = info[0]
            start_frame = info[1]
            end_frame = info[2]
            pred_output = highest_output[idx]
            pred_label = idx_to_class[pred_output.item()]
            ground_truth = idx_to_class[og_label_output[idx].item()]
            dog_name = dog_id[idx].split('/')[-1]
            pred = [dog_name, start_frame[idx].item(), end_frame[idx].item(), pred_label, ground_truth]
            #print("pred", pred)
            predictions.append(pred)

             
        #print("predictions", predictions)
        eval_this = eval_metric(output, label)
        eval_meter.update(eval_this.item(), data.shape[0])
         
    return eval_meter.avg, predictions

def _train_batch(model, data, label, criterion, optimizer):
        optimizer.zero_grad()
        output = model(data)
        loss_this = criterion(output, label)
        loss_this.backward()
        optimizer.step()
        return loss_this.item()

# def _train_epoch(model, epoch, train_loader, scheduler, optimizer, criterion, device):
#     model.train()
#     loss_meter = AverageMeter()
#     start_time = time.time()
#     transforms = get_train_transforms()
#     images = list()
#     for image_paths, texts, labels in train_loader:
#         for img in range(0,len(image_paths)):
#             image = [Image.open(image_paths[img]).convert("RGB")]
#             images.extend(image)
#         print("img size before", len(images)) #this is 10, so it's in line with the batch size
#         images = transforms(images) 
#         print("img size after", images.size()) # torch.Size([30, 224, 224]) --> zelfde voor animal kingdom
#         images = images.view((8, -1) + images.size()[-2:]) # 8 is batch_size
#         print("image after view", images.size()) # [10, 3, 224, 224] mijn dataset --> animalkingdom [10, 3, 224, 224]
#         num_encoded_labels = [class_to_idx[label] for label in labels]
#         labels = torch.tensor(num_encoded_labels, dtype=torch.long, device=device)
#         print("labels inhoud", labels)
#         print("labels", labels.size()) #labels heeft size 10, in animalkingdom --> [8,140] (daar is de batchsize 8)
#         loss_this = _train_batch(model, images, labels, criterion, optimizer)
#         loss_meter.update(loss_this, image_paths.shape[0])
#     elapsed_time = time.time() - start_time
#     scheduler.step()
#     print("Epoch [" + str(epoch + 1) + "]"
#             + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
#             + " loss: " + "{:.4f}".format(loss_meter.avg), flush=True)

def _train_epoch(model, epoch, train_loader, scheduler, optimizer, criterion, device):
        model.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        for data, label, info in tqdm(train_loader, desc=f"training", unit="batch"):
            data, label = data.to(device), label.to(device)
            # print("data in train_epoch", data.size())
            # print("label inhoud", label)
            # print("label in train_epoch", label.size())
            loss_this = _train_batch(model, data, label, criterion, optimizer)
            loss_meter.update(loss_this, data.shape[0])
        elapsed_time = time.time() - start_time
        scheduler.step()

        print("Epoch [" + str(epoch + 1) + "]"
                + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                + " loss: " + "{:.4f}".format(loss_meter.avg), flush=True)
    
def train(model, start_epoch, end_epoch, train_loader, scheduler, optimizer, criterion, device):
    best_model = model
    best_map = 0
    best_epoch = 0
    preds = []
    best_preds = []
    for epoch in range(start_epoch, end_epoch):
        print(f"this is epoch {epoch+1}")
        _train_epoch(model, epoch, train_loader, scheduler, optimizer, criterion, device)
        if (epoch + 1) % 5 == 0:
                res, preds = test(model, test_loader, device)
                curr_map = res * 100
                print("[INFO] Test MAP: {:.2f}".format(res * 100), flush=True)

                print("saving checkpoint")
                save(model, epoch)

                if curr_map > best_map:
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                    best_map = curr_map
                    best_preds = preds

    print("best preds", best_preds)
    print("preds train", preds)
    print("saving best model..")
    torch.save(best_model, 'best_transfer_model.pt')
    torch.save(best_model.state_dict(), 'best_transfer_statedict.pt')
    save_to_csv(best_preds, r"/home/albert/Lisanne/MSQNet/multi-label-action-main/predictions")
    print(f"best map was {best_map} at epoch {best_epoch + 1}")

    print("saving last model..")
    torch.save(model, 'last_transfer_model.pt')
    torch.save(model.state_dict(), 'last_transfer_statedict.pt')
    save_to_csv(preds, r"/home/albert/Lisanne/MSQNet/multi-label-action-main/predictions_last")
    print("done saving")

# Prepare the data for training
# Load possible prompts
# Initialize custom dataset
# Initialize dataloaders
dog_names_train = ['binnur_side_Trim', 'binnur_front_trim', 'Arwen -OFT', 'unal_front_Trim', 'unal_side_trim',
                   'Akrep_OFT_Cam1_Trim', 'Alfa_OFT_Cam1_Trim', 'Alisa_OFT_Cam1_Trim', 'Aria_OFT_ Cam1_Trim']
prompts = {
    'standing still': ['the dog is standing still', 'the dog is standing still inside the room'],
    'walking': ['the dog is walking', 'the dog is walking through the room'],
    'trotting': ['the dog is trotting', 'the dog is trotting through the room'],
    'circling': ['The dog is walking in circles in the room', 'the dog is turning in circles', 
                 'the dog is circling through the room'],
    'jumping up': ['the dog is jumping up', 'the dog is jumping against the wall'],
    'sitting': ['the dog is sitting', 'the dog is sitting in a room on the ground'],
    'sitting abnormally': ['The dog is sitting on the ground and one of its legs is sticking out in an abnormal way',
                           'The dog is sitting abnormally',
                           'The dog is sitting with its legs not completely under its body'],
    'lying down': ['the dog is lying down', 'the dog is lying down in a room'],
    'obscured': ['the dog is hidden for the camera', 'the dog is not visible', 'the dog is obscured']
    }

class_to_idx = {
    'standing still': 0,
    'walking': 1,
    'trotting': 2,
    'circling': 3,
    'jumping up': 4,
    'sitting': 5,
    'sitting abnormally': 6,
    'lying down': 7,
    'obscured': 8
    }

idx_to_class = {
    0: 'standing still',
    1: 'walking',
    2: 'trotting',
    3: 'circling',
    4: 'jumping up',
    5: 'sitting',
    6: 'sitting abnormally',
    7: 'lying down',
    8: 'obscured'
    }

# Create datasets and dataloaders
# data_path = r"C:\Thesis\deepaction_dog\dog_annotation2\frames\spatial" #own pac
data_path = r"/home/albert/Lisanne/frames/pain/frames/spatial"
print("creating train dataset...")
train_dataset = Dog(data_path, class_to_idx, 10, transform=get_train_transforms(), mode='train')
print("creating test dataset...")
test_dataset = Dog(data_path, class_to_idx, 10, transform=get_test_transforms(), mode='test')
print("finished creating datasets")

print("creating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True, drop_last=True)
print("finished creating DataLoaders")

#Initialize model
print("initializing msqnet model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_path = r"C:\Thesis\dog_pain_lisanne\clip_action_prediction\Dog behavior prediction with CLIP\models\best_pretrained_model.pt"
model_path = r"/home/albert/Lisanne/MSQNet/multi-label-action-main/model_trains/best_pretrained_model.pt"
model = torch.load(model_path, map_location=device)
model.to(device)

# Freeze the parameters of the model, this can be set to True if you want to fully fine-tune the model
for param in model.parameters():
    param.requires_grad = False

# Replace final layer with layer that matches the number of classes, this will have requires_grad = True automatically
class_list = list(class_to_idx.keys())
class_embed = _get_text_features(class_list)
num_classes, embed_dim = class_embed.shape
model.num_classes = num_classes
model.query_embed = nn.Parameter(class_embed)
model.group_linear = GroupWiseLinear(9, embed_dim, bias=True)
print("done initializing")

#Initialize loss, optimizer and scheduler
loss = nn.CrossEntropyLoss() #cross entropy because there are multiple classes 
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #parameters from CLIP paper
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

#Train the model
print("training the model..")
model.to(device)
train(model, 0, 100, train_loader, scheduler, optimizer, loss, device)

print("finished training")

#Test the model
# print("testing the model..")
# test(model, test_loader, device, True)
# print("finished testing")