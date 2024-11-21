import time
import math
import torch
import copy
import pandas as pd
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F
from utils.utils import AverageMeter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import TimesformerModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel, logging
from tqdm import tqdm
from scipy.ndimage import zoom
from PIL import Image


class TimeSformerCLIPInitVideoGuide(nn.Module):
    def __init__(self, class_embed, num_frames):
        super().__init__()
        self.num_classes, self.embed_dim = class_embed.shape
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", num_frames=num_frames, ignore_mismatched_sizes=True)
        self.linear1 = nn.Linear(in_features=self.backbone.config.hidden_size, out_features=self.embed_dim, bias=False)
        self.pos_encod = PositionalEncoding(d_model=self.embed_dim)
        self.image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.linear2 = nn.Linear(in_features=self.backbone.config.hidden_size + self.embed_dim, out_features=self.embed_dim, bias=False)
        self.query_embed = nn.Parameter(class_embed)
        self.transformer = nn.Transformer(d_model=self.embed_dim, batch_first=True)
        self.group_linear = GroupWiseLinear(self.num_classes, self.embed_dim, bias=True)

    def forward(self, images):
        # print("forward", images.size())
        b, t, c, h, w = images.size()

        x = self.backbone(images)[0]
        x = self.linear1(F.adaptive_avg_pool1d(x.transpose(1, 2), t).transpose(1, 2))
        x = self.pos_encod(x)
        image_output = self.image_model(images.reshape(b*t, c, h, w), output_attentions=True)
        attn_weights = image_output.attentions
        video_features = self.image_model(images.reshape(b*t, c, h, w))[1].reshape(b, t, -1).mean(dim=1, keepdim=True)
        self.query_embed.to('cuda')
        video_features.to('cuda')
        # print(f"device of video features: {video_features.get_device()}")
        # print(f"device of query embed: {self.query_embed.get_device()}")
        # print(f"video_features before reshape: {video_features.shape}")
        # print(f"query features before reshape: {self.query_embed.shape}")
        query_embed = self.linear2(torch.concat((self.query_embed.unsqueeze(0).repeat(b, 1, 1), video_features.repeat(1, self.num_classes, 1)), 2))
        # print(f"video_features after reshape: {video_features.repeat(1, self.num_classes, 1).shape}")
        # print(f"query features after reshape: {self.query_embed.unsqueeze(0).repeat(b, 1, 1).shape}")
        hs = self.transformer(x, query_embed) # b, t, d
        out = self.group_linear(hs)
        return out, attn_weights
    

class TimeSformerCLIPInitVideoGuideExecutor:
    def __init__(self, train_loader, test_loader, criterion, eval_metric, class_list, 
                 test_every, distributed, gpu_id, unsupervised, idle, one_action, second_dimension) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(gpu_id)
        self.eval_metric = eval_metric.to(gpu_id)
        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.gpu_id = gpu_id
        self.unsupervised = unsupervised
        self.idle = idle
        self.one_action = one_action
        self.second_dimension = second_dimension
        self.progress_record_path = r"/home/albert/Lisanne/MSQNet/multi-label-action-main/checkpoints/progression_files"
        num_frames = self.train_loader.dataset[0][0].shape[0]
        logging.set_verbosity_error()
        class_embed = self._get_text_features(class_list)
        print(f"this is the action dict: {class_list}")
        model = TimeSformerCLIPInitVideoGuide(class_embed, num_frames).to(gpu_id)

        if distributed: 
            self.model = DDP(model, device_ids=[gpu_id])
        else: 
            self.model = model

        for p in self.model.parameters():
            p.requires_grad = True #This might affect the transfer learning
        for p in self.model.image_model.parameters():
            p.requires_grad = False
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": 0.00001}])#0.00001 original lr, orignal weight decay is 0
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

    #@staticmethod
    def _get_prompt(self, cl_names, prompt_dict):
        temp_prompt = []
        print(f"this is the prompt dict: {prompt_dict}")
        for c in cl_names:
            print(f"this is c: {c}")
            # # the code for multiple prompts
            # prompt = random.choice(prompt_dict[c])
            if self.second_dimension:
                prompt = prompt_dict[c]
            else:
                if c == 'standing still':
                    prompt = f"a dog is standing on four feet inside a room"
                elif c == 'not standing still':
                    prompt = f"a dog is not standing on four feet inside a room"
                else:
                # the code for a dog is {label} in a room
                    prompt = f"a dog is {c} in a room"
            temp_prompt.append(prompt)
        print("temp prompts", temp_prompt)
        return temp_prompt
    
    def _get_prompt_dict(self):
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
        return prompts
    
    def _get_prompt_dict_idle(self):
        prompts = {
            'standing still': ['the dog is standing still', 'the dog is standing still inside the room'],
            'sitting': ['the dog is sitting', 'the dog is sitting in a room on the ground'],
            'sitting abnormally': ['The dog is sitting on the ground and one of its legs is sticking out in an abnormal way',
                                'The dog is sitting abnormally',
                                'The dog is sitting with its legs not completely under its body'],
            'lying down': ['the dog is lying down', 'the dog is lying down in a room'],
            }
        return prompts
    

    def _get_prompt_dict_second_dimension(self):
        prompts = {
            'decreased weight bearing': 'a dog is not putting weight on one of its paws',
            'normal weight bearing': 'a dog is putting even weight on all of its paws',
            'rigid posture': 'the dog has a stiff posture',
            'rigid gait': 'the dog is moving in a stiff way',
            'no rigidness': 'the dog does not look stiff',
            'ul': ''
        } 
        return prompts

    
    def _get_class_idx(self):
        dict = {
            'standing still': 0,
            'walking': 1,
            'trotting': 2,
            'circling': 3,
            'jumping up': 4,
            'sitting': 5,
            'sitting abnormally': 6,
            'lying down': 7,
            'obscured': 8,
            'UL': 9
            }
        return dict

    def _get_idx_class(self):
        dict = {
            0: 'standing still',
            1: 'walking',
            2: 'trotting',
            3: 'circling',
            4: 'jumping up',
            5: 'sitting',
            6: 'sitting abnormally',
            7: 'lying down',
            8: 'obscured',
            9: 'UL'
            }
        return dict
    
    def _get_class_idx_idle(self):
        dict = {
            'standing still': 0,
            'sitting': 1,
            'sitting abnormally': 2,
            'lying down': 3,
            'UL': 4
            }
        return dict

    def _get_idx_class_idle(self):
        dict = {
            0: 'standing still',
            1: 'sitting',
            2: 'sitting abnormally',
            3: 'lying down',
            4: 'UL'
            }
        return 
    
    def _get_idx_class_second_dimension(self):
        prompts = {
            0: 'decreased weight bearing',
            1: 'normal weight bearing',
            2: 'rigid posture',
            3: 'rigid gait',
            4: 'no rigidness',
            5: 'ul',
        } 
        return prompts
    
    def _generate_idx_class(self):
        dict = {
            0: self.one_action,
            1: "not " + self.one_action,
            2: "UL"
        }
        return dict
    
    
    def _get_text_features(self, cl_names):
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        if self.idle:
            prompts = self._get_prompt_dict_idle()
        elif self.second_dimension:
            prompts = self._get_prompt_dict_second_dimension()
        else:
            prompts = self._get_prompt_dict()
        act_prompt = self._get_prompt(cl_names, prompts)
        texts = tokenizer(cl_names, padding=True, return_tensors="pt")
        text_class = text_model(**texts).pooler_output.detach()
        return text_class

    def _train_batch(self, data, label, img_names):
        self.optimizer.zero_grad()
        data.to(self.gpu_id), label.to(self.gpu_id)
        output, attn_weights = self.model(data)


        loss_this = self.criterion(output, label)
        loss_this.backward()


        eval_res = self.eval_metric(output, label.long().to(self.gpu_id))
        #print(f"label: {label.shape}")
        #self.overlay_heatmap_on_image(attn_weights, data, img_names, layer_idx=0, head_idx=0)

        # print(f"the train data shape is: {data.shape}")
        # print(f"the train output is: {output}")
        # print(f"the train label is: {label}")
        # print(f"the train loss is: {loss_this.item()}")
        # print(f"the train eval is: {eval_res.item()}")

        if(math.isnan(loss_this.item()) == False):
            # print("loss is not NaN")
            ...
        elif(math.isnan(loss_this.item())):
            for name, param in self.model.named_parameters():
                print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
                pos_inf = torch.max(param) 
                neg_inf = torch.min(param) 
                print(f"max value is {pos_inf} and min value is {neg_inf}")

        self.optimizer.step()
        return loss_this.item(), eval_res.item()

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        eval_meter = AverageMeter()
        start_time = time.time()
        for (data, label, info, img_names) in tqdm(self.train_loader, desc=f"training", unit="batch"):
            if(data.shape[0] == 0):
                print("data shape is zero, skips this instance")
                continue
            data, label = data.to(self.gpu_id, non_blocking=True), label.to(self.gpu_id, non_blocking=True)
            # print("data in train_epoch", data.size())
            # print("label inhoud", label)
            # print("label in train_epoch", label.size())
            # print(f"img names in train epoch: {len(img_names[0])}")
            loss_this, eval_res = self._train_batch(data, label, img_names)
            eval_meter.update(eval_res, data.shape[0])
            loss_meter.update(loss_this, data.shape[0])
        elapsed_time = time.time() - start_time
        self.scheduler.step()
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            print("Epoch [" + str(epoch + 1) + "]"
                  + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                  + " loss: " + "{:.4f}".format(loss_meter.avg)
                  + " mAP: " + "{:.4f}".format(eval_meter.avg * 100), flush=True)
        self.update_progress_record(loss_meter.avg, eval_meter.avg, epoch, self.progress_record_path, 'train')
        return loss_this
    
    def train(self, start_epoch, end_epoch):
        best_model = self.model
        best_map = 0
        best_epoch = 0
        preds = []
        for epoch in range(start_epoch, end_epoch):
            print(f"start epoch {epoch}")
            loss = self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                eval_test, preds = self.test_supervised()
                curr_map = eval_test * 100
                
                print(f"the test loss is: {loss}")

                print("saving checkpoint")
                #self.save(epoch)
                self.update_progress_record(loss, curr_map, epoch, self.progress_record_path, 'val')
                self.save_to_csv(preds, r"/home/albert/Lisanne/MSQNet/multi-label-action-main/predictions_last", "", epoch)


                if curr_map > best_map:
                    best_model = copy.deepcopy(self.model) 
                    best_map = curr_map
                    best_epoch = epoch    
                    best_preds = preds  
                if (self.distributed and self.gpu_id == 0) or not self.distributed:
                    print("[INFO] Evaluation Metric: {:.2f}".format(eval_test * 100), flush=True)

        print("saving best model..")
        torch.save(best_model.state_dict(), 'best_scratch_statedict_run2.pt')
        torch.save(best_model, 'best_scratch_model_run2.pt')
        self.save_to_csv(best_preds, r"/home/albert/Lisanne/MSQNet/multi-label-action-main/predictions", "best", end_epoch)

        print("saving last model..")
        torch.save(self.model.state_dict(), 'last_scratch_statedict_run2.pt')
        torch.save(self.model, 'last_scratch_model_run2.pt')
        print("done saving")
        print(f"best model at epoch {best_epoch} with map {best_map}")
                
    
    # def test(self):
    #     self.model.eval()
    #     eval_meter = AverageMeter()
    #     for data, label in self.test_loader:
    #         data, label = data.to(self.gpu_id), label.long().to(self.gpu_id)
    #         with torch.no_grad():
    #             output = self.model(data)
    #         print("output", output)
    #         print("label", label)
    #         eval_this = self.eval_metric(output, label)
    #         eval_meter.update(eval_this.item(), data.shape[0])
    #     return eval_meter.avg
    
    def test_supervised(self):
        self.model.eval()
        eval_meter = AverageMeter()
        loss_meter = AverageMeter()
        if self.idle:
            idx_to_class = self._get_idx_class_idle()
        elif self.one_action:
            idx_to_class = self._generate_idx_class()
            print(f"this is the idx to class: {idx_to_class}")
        elif self.second_dimension:
            idx_to_class = self._get_idx_class_second_dimension()
            print(f"this is the idx to for second dimension: {idx_to_class}")
        else:
            idx_to_class = self._get_idx_class()
            
        predictions = [] # col 1 has dog id, col 2 start frame, col 3 end frame, col 4 predicted label, col 4 ground truth
        for (data, label, info, img_names) in self.test_loader:
            data, label = data.to(self.gpu_id), label.long().to(self.gpu_id)
            with torch.no_grad():
                output, attn_weights = self.model(data)
                #loss_this = self.criterion(output.long().to(self.gpu_id), label.long().to(self.gpu_id))
            highest_output = torch.argmax(output, dim=1) # for single label classification
            og_label_output = torch.argmax(label, dim=1)
            output_array = output.cpu().numpy()

            self.overlay_heatmap_on_image(attn_weights, data, img_names, layer_idx=0, head_idx=0)
            print(f"the data size: {data.shape}")
            print(f"the label: {label}")
            print(f"the label size: {label.shape}")
            print(f"info: {info}")
            print(f"highest_output: {highest_output}")
            print(f"og label output: {og_label_output}")
            print(f"output size: {output.shape}")
            print(f"output: {output}")

            for idx in range(label.shape[0]):
                info_instance = info[idx]
                dog_id = info_instance[0]
                start_frame = info_instance[1]
                end_frame = info_instance[2]
                highest_pred_output = highest_output[idx]
                highest_pred_label = idx_to_class[highest_pred_output.item()]
                ground_truth = idx_to_class[og_label_output[idx].item()]
                dog_name = dog_id.split('/')[-1]
                pred = [dog_name, start_frame, end_frame, output_array[idx], highest_pred_label, ground_truth]
                print("pred", pred)
                predictions.append(pred)

                
            #print("predictions", predictions)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
            # print(f"the test eval is: {eval_this.item()}")
            # print(f"the test data shape is: {data.shape}")
            # print(f"the test output is: {output}")
            # print(f"the test label is: {label}")
            # print(f"the test loss is: {loss_this.item()}")

            
        return eval_meter.avg, predictions
    
    def test_unsupervised(self):
        self.model.eval()
        if self.idle:
            idx_to_class = self._get_idx_class_idle()
        elif self.one_action:
            idx_to_class = self._generate_idx_class()
            print(f"this is the idx to class: {idx_to_class}")
        else:
            idx_to_class = self._get_idx_class()
        # col 1 has dog id, col 2 start frame, col 3 end frame, col 4 predicted label, col 4 ground truth (unknown in case of unsupervised)
        predictions = [] 
        for data, info in self.test_loader: #TODO check if unpacking here goes correct
            data  = data.to(self.gpu_id)
            with torch.no_grad():
                output = self.model(data)
                #loss_this = self.criterion(output.long().to(self.gpu_id), label.long().to(self.gpu_id))
            highest_output = torch.argmax(output, dim=1) # for single label classification
            output_array = output.cpu().numpy()
            # print("output", output)
            # print("output_array", output_array)
            # print("highest output", highest_output)

            for idx in range(data.shape[0]):
                dog_id = info[0]
                start_frame = info[1]
                end_frame = info[2]
                highest_pred_output = highest_output[idx]
                highest_pred_label = idx_to_class[highest_pred_output.item()]
                ground_truth = "unknown"
                dog_name = dog_id[idx].split('/')[-1]
                pred = [dog_name, start_frame[idx].item(), end_frame[idx].item(), output_array[idx], highest_pred_label, ground_truth]
                predictions.append(pred)
            
        return predictions
    
    def save_to_csv(self, predictions, path, text, epoch):
     df = pd.DataFrame(predictions, columns=['Dog_id', 'Start frame', 'End frame', 'Output array', 'Prediction', 'Ground truth'])
     sorted_on_id = df.groupby('Dog_id')
    #  print("preds", predictions)
    #  print("sorted", sorted_on_id)

     for dog_path, dog_df in sorted_on_id:
          dog_split = dog_path.split('/')
          dog_name = dog_split[-1]
          file_name = f"{text}epoch-{epoch}-predictions-{dog_name}.csv"
          output = os.path.join(path, file_name)
          dog_df.to_csv(output, index=False)
          print(f"saved to {output}")

    def save(self, epoch):
        # backbone_state_dict = self.model.backbone.state_dict()
        # linear_state_dict = self.model.linear1.state_dict()
        # transformer_state_dict = self.model.transformer.state_dict()
        # query_embed_state_dict = self.model.query_embed.state_dict()
        # group_linear_state_dict = self.model.fc.state_dict()
        # optimizer_state_dict = self.optimizer.state_dict()
        torch.save(self.model.state_dict(), f"./checkpoints/state_dict_epoch{epoch}")
        torch.save(self.model, f"./checkpoints/model_epoch{epoch}")
    
    def update_progress_record(self, loss, metric, epoch, path, mode):
        if mode == 'train' or mode == 'val':
            filename = os.path.join(path, f"{mode}_progress.csv")
        else:
            print(f"{mode} is not a valid command for the mode parameter of update_progress_record")
        info = [[mode, epoch, round(loss,4), round(metric,4)]]
        df = pd.DataFrame(info, columns=['type', 'epoch', 'loss', 'mAP'])

        if not os.path.isfile(filename): #intialize csv file
            print(f"initialize {filename} for progress")
            df.to_csv(filename, index=False)
        else:  # else it exists so append without writing the header
            print(f"append progress to {filename}")
            df.to_csv(filename, mode='a', header=False, index=False)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.backbone.load_state_dict(checkpoint["backbone"])
        self.model.linear.load_state_dict(checkpoint["linear"])
        self.model.transformer.load_state_dict(checkpoint["transformer"])
        self.model.query_embed.load_state_dict(checkpoint["query_embed"])
        self.model.group_linear.load_state_dict(checkpoint["group_linear"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def overlay_heatmap_on_image(self, attention_weights, images, img_names, layer_idx=0, head_idx=0):
        # Extract the attention weights for a specific layer and head
        # print(f"img names shape: {len(img_names)}")
        # print(f"images shape: {images.shape}")
        # print(f"the img names: {len(img_names)}")

        # print(f"attention weights: {attention_weights}")
        # print(f"attention weights size: {attention_weights[0]}")
        # print(f"attention weights shape: {attention_weights[0][0].shape}")
        attn = attention_weights[layer_idx][head_idx].detach().cpu().numpy()
        
        # Get the first frame of the first image in the batch
        image = images[0, 0].permute(1, 2, 0).detach().cpu().numpy()
        img_name = img_names[0][0]
        dog_name = img_name.split('/')[-1]

        print(f"img_name: {dog_name}")
        # Check the dimensions of the attention map and handle appropriately
        print(f"attn shape: {attn.shape}")
        attn = attn.mean(axis=0)

        # Assuming attn has shape (197, 197) and we need to map it to the image shape (224, 224)
        # Exclude the first token (class token) and reshape the attention map to (14, 14)
        num_patches = int(np.sqrt(attn.shape[-1] - 1))
        attn = attn[:, 1:].reshape(attn.shape[0], num_patches, num_patches)

        print(f"attn shape: {attn.shape}")
        # Resize the attention weights to match the dimensions of the image
        attn_resized = zoom(attn, (1, image.shape[0] / num_patches, image.shape[1] / num_patches), order=1)
        attn_resized = attn_resized.mean(axis=0)  # Average over the attention heads if there are multiple

        # Normalize the attention weights
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())

        #Plot the image
        print(f"attn_resized: {attn_resized.shape}")
        background_image = Image.open(img_name)
        background_resized = background_image.resize((224,224))
        plt.imshow(background_resized)
        
        # Overlay the heatmap
        sns.heatmap(attn_resized, cmap='viridis', alpha=0.1, annot=False, cbar=False, zorder=2)
        
        plt.axis('off')
    
        plt.savefig(f"/home/albert/Lisanne/MSQNet/multi-label-action-main/heatmaps/{dog_name.split('.')[0]}.png", bbox_inches='tight', pad_inches=0)
        plt.show()


class PositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        out = self.dropout(x)
        return out


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