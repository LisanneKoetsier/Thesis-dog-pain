from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from tqdm import tqdm
import numpy as np
import pprint
import copy
import joblib
import graphviz
import os
import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn

import lib.model.two_stream as model_set
from lib.data_build import Data_loader
from lib.config_file import cfg
import lib.utils.logging as log
import lib.utils.checkpoint as cu
from lib.utils.meter import Train_meter, Val_meter
import lib.utils.solver as sol

class EarlyStopper:
    """from stackoverflow: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    Patience is the number of episodes it is guaranteed to run before it stops"""

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print("loss is decreasing", validation_loss, self.min_validation_loss)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print("loss is increasing", validation_loss, self.min_validation_loss)
            if self.counter >= self.patience:
                print("early stop!")
                return True
        return False


logger = log.get_logger(__name__)
def build_model(cfg, decision_tree):
    
    if cfg.MODEL.TYPE == "two_stream" or cfg.MODEL.TYPE == "ethogram" or cfg.MODEL.TYPE == "ethogram_only":
        model = eval(f"model_set.{cfg.MODEL.MODEL_NAME}(cfg, decision_tree).cuda()")
    else:
        model = eval(f"model_set.{cfg.MODEL.MODEL_NAME}(cfg).cuda()")
    print(f"this is the model: {model}")
    if cfg.SOLVER.METHOD == "sgd":
        optim = torch.optim.SGD(
            model.parameters(), 
            lr=cfg.SOLVER.BASE_LR, 
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
            )
    elif cfg.SOLVER.METHOD == "adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR, 
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.METHOD == "adadelta":
        optim = torch.optim.Adadelta(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            rho=0.95,
            eps=1e-07,
        )
    
    train_meter = Train_meter(cfg)
    test_meter = Val_meter(cfg) 
    return (
        model,
        optim,
        train_meter,
        test_meter,
    )
        

@torch.no_grad()   
def val_epoch(
    cfg, 
    model, 
    test_loader, 
    test_meter, 
    cur_epoch,
    data_type,
    loss_pack, 
    loss_dict=None
    ):

    model.eval()
    if loss_dict is not None:
        awl = sol.AutomaticWeightedLoss(2).cuda()
        awl.load_state_dict(loss_dict)
    test_meter.time_start()
    for cur_iter, (inputs, labels, ethogram, start_name) in enumerate(tqdm(test_loader)):
        test_meter.time_pause()
        test_meter.update_data()
        test_meter.time_start()
        print(f"this is the ethogram: {ethogram}")
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        
        if isinstance(labels, list):
            labels[0] = (labels[0].unsqueeze(-1)).cuda()
            labels[1] = labels[1].cuda()
        else:
            labels = labels.cuda()
            if data_type == "simple":
                labels = labels.unsqueeze(-1)

        if cfg.MODEL.TYPE == "ethogram":
            ethogram_normalized = copy.deepcopy(ethogram)
            ethogram_normalized[ethogram_normalized>1] = 1 #TODO uncomment
            # print(f"this is the val normalized ethogram: {ethogram_normalized}")
            # print(f"this is the val og ethogram: {ethogram}") 
            # print(f"this is the ethogram normalized: {ethogram_normalized}")
            outputs = model(inputs, ethogram_normalized.cuda()) #REMINDER: changed to normal ethogram instead of normalized to test
        elif cfg.MODEL.TYPE == "ethogram_only":
            ethogram_normalized = copy.deepcopy(ethogram)
            ethogram_normalized[ethogram_normalized>1] = 1 #TODO: uncomment
            # print(f"this is the ethogram normalized: {ethogram_normalized}")
            # print(f"this is the train normalized ethogram: {ethogram_normalized}")
            # print(f"this is the train og ethogram: {ethogram}")
            outputs = model(ethogram_normalized.cuda()) #REMINDER: changed to normal ethogram to test
        else:
            outputs = model(inputs, [])

        if isinstance(outputs, list):
            out1, out2 = outputs[0], outputs[1]
            loss_func1 = loss_pack[0]
            loss_func2 = loss_pack[1]
            
            acc1 = sol.get_binary_acc(out1, labels[0])
            acc2 = sol.get_accuracy(out2, labels[1])
            loss1 = loss_func1(out1, labels[0].float())
            loss2 = loss_func2(out2, labels[1])
            loss = awl(loss1, loss2)
        else:
            if data_type == "simple":
                loss_func = loss_pack[0]
                loss = loss_func(outputs, labels.float())
                acc = sol.get_binary_acc(outputs, labels)
            else:
                loss_func = loss_pack[1]
                loss = loss_func(outputs, labels)
                acc = sol.get_accuracy(outputs, labels)


        if cfg.MODEL.TYPE == "ethogram_only":
            #print(f"ethogram: {ethogram.shape[0]}")
            batch_size = ethogram.shape[0]
            #print(f"batch size: {batch_size}")
        else:
            batch_size = inputs[0].size(0)
            
        if data_type in ["simple", "diff"]:
            loss, acc = (
                loss.item(),
                acc.item(),
            )
            test_meter.update_states(batch_size, loss=loss, acc=acc)
        else:
            test_meter.update_states(
                batch_size,
                
                loss = loss.item(),
                loss1 = loss1.item(),
                loss2 = loss2.item(),
                acc1 = acc1.item(),
                acc2 = acc2.item(),
            )
        
        test_meter.time_pause()
        test_meter.update_batch()
        test_meter.time_start()
    
    test_meter.update_epoch(cur_epoch)
    if data_type in ["simple", "diff"]:
        accuracy_score = test_meter.info["acc"].avg
        loss_score = test_meter.info["loss"].avg #check if this needs to be average
    else:
        accuracy_score =test_meter.info["acc1"].avg
        loss_score = test_meter.info["loss1"] #need to check this, idk if this is the right loss 
    test_meter.reset()
    return accuracy_score, loss_score

def train_epoch(
    cfg, 
    model, 
    optim, 
    train_loader, 
    train_meter,
    cur_epoch, 
    data_type,
    loss_pack,
    ):
    model.train()
    data_size = len(train_loader)
    train_meter.time_start()
    for cur_iter, (inputs, labels, ethogram, start_name) in enumerate(tqdm(train_loader)):
        train_meter.time_pause()
        train_meter.update_data()
        train_meter.time_start()
        
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
            
        lr = sol.get_lr_at_epoch(cfg, cur_epoch+float(cur_iter)/data_size)
        sol.set_lr(optim, lr)
        if isinstance(labels, list):
            labels[0] = (labels[0].unsqueeze(-1)).cuda()
            labels[1] = labels[1].cuda()
        else:
            labels = labels.cuda()
            if data_type == "simple":
                labels = labels.unsqueeze(-1) 
                
        #print(f"this is the ethogram: {ethogram}")
        if cfg.MODEL.TYPE == "ethogram":
            #(f"these are the inputs: {inputs}")
            # print(f"ethogram before normalization: {ethogram}")
            ethogram_normalized = copy.deepcopy(ethogram)
            ethogram_normalized[ethogram_normalized>1] = 1 #TODO uncomment
            #print(f"this is the ethogram normalized: {ethogram_normalized}")
            # print(f"this is the train normalized ethogram: {ethogram_normalized}")
            # print(f"this is the train og ethogram: {ethogram}")
            outputs = model(inputs, ethogram_normalized.cuda()) #REMINDER: changed to normal ethogram to test
        elif cfg.MODEL.TYPE == "ethogram_only":
            #print(f"ethogram: {ethogram}")
            ethogram_normalized = copy.deepcopy(ethogram)
            ethogram_normalized[ethogram_normalized>1] = 1 #TODO: uncomment
            #print(f"this is the ethogram normalized: {ethogram_normalized}")
            # print(f"this is the train normalized ethogram: {ethogram_normalized}")
            # print(f"this is the train og ethogram: {ethogram}")
            outputs = model(ethogram_normalized.cuda()) #REMINDER: changed to normal ethogram to test
        else:
            outputs = model(inputs, [])

        if isinstance(outputs, list):
            out1, out2 = outputs[0], outputs[1]
            loss_func1 = loss_pack[0]
            loss_func2 = loss_pack[1]
            awl = sol.AutomaticWeightedLoss(2).cuda()

            acc1 = sol.get_binary_acc(out1, labels[0])
            acc2 = sol.get_accuracy(out2, labels[1])
            loss1 = loss_func1(out1, labels[0].float())
            loss2 = loss_func2(out2, labels[1])
            loss = awl(loss1, loss2)
        else:
            if data_type == "simple":
                loss_func = loss_pack[0]
                loss = loss_func(outputs, labels.float())
                acc = sol.get_binary_acc(outputs, labels)
            else:
                loss_func = loss_pack[1]
                loss = loss_func(outputs, labels)
                acc = sol.get_accuracy(outputs, labels)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        if cfg.MODEL.TYPE == "ethogram_only":
            #print(f"ethogram: {ethogram.shape[0]}")
            batch_size = ethogram.shape[0]
            #print(f"batch size: {batch_size}")
        else:
            batch_size = inputs[0].size(0)
        if data_type in ["simple", "diff"]:
            loss, acc = (
                loss.item(),
                acc.item(),
            )
            train_meter.update_states(batch_size, lr, loss=loss, acc=acc)
        else:
            
            train_meter.update_states(
                batch_size,
                lr,
                loss = loss.item(),
                loss1 = loss1.item(),
                loss2 = loss2.item(),
                acc1 = acc1.item(),
                acc2 = acc2.item(),
            )

        
        train_meter.time_pause()
        train_meter.update_batch()
        train_meter.time_start()
        torch.cuda.empty_cache()
    
    train_meter.update_epoch(cur_epoch)
    train_meter.reset()
    return awl.state_dict() if data_type == "aux" else None

def train_net_only(cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)

    log.setup_logging(cfg.OUT_DIR, cfg.CHECKPOINTS_FOLD)
    logger.info(pprint.pformat(cfg))

    data_container = Data_loader(cfg)
    logger.info("start load dataset")
    train_loader, loss_wtrain = data_container.construct_loader("train")
    test_loader, loss_wtest = data_container.construct_loader("test")

    rf_classifier = None
    if cfg.MODEL.DECISION_TREE:
        
        data_set_loader = train_loader.dataset
        ethogram_list, ethogram_labels = data_set_loader._get_entire_ethogram_set()
        print(f"this is the ethogram list from the dataset: {len(ethogram_list)}")
        #print(f"this is the ethogram list from the dataset: {ethogram_list}")
        ethogram_data = np.array(ethogram_list)
        ethogram_data_boolean = (ethogram_data > 0)
        print(f"this is ethogram data shape")
        print(f"this is ethogram data: {ethogram_data}")

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_leaf_nodes=10)
        rf_classifier.fit(ethogram_data_boolean, ethogram_labels)
        decision_tree_save = os.path.join(cfg.CHECKPOINTS_FOLD, 'random_forest_model.joblib')
        joblib.dump(rf_classifier, decision_tree_save)

        ethogram_feature_names = ["standing still", "walking", "trotting", "circling", "jumping up", "sitting", "sitting abnormally", "lying down", "obscured", "UL", "decreased weight bearing"]
        for i, tree in enumerate(rf_classifier.estimators_):
            dot_data = export_graphviz(tree, out_file=None,
                                       feature_names=ethogram_feature_names,
                                       class_names=["no pain", "pain"], 
                                       filled=True, rounded=True,  
                                       special_characters=True, 
                                       impurity=False, 
                                       proportion=False,
                                       label='root',
                                       precision=2)
            
            dot_data = dot_data.replace('≤ 0.5', 'is False').replace('> 0.5', 'is True')
            # Use graphviz to render the DOT file
            graph = graphviz.Source(dot_data)
            file_path = os.path.join("decision_trees", f"decision_tree_{i}.pdf")
            graph.render(file_path)  # Save to file
            print(f"Decision tree {i} saved to {file_path}")

    (
        model,
        optim,
        train_meter,
        test_meter,
    ) = build_model(cfg, rf_classifier)

    start_epoch = cu.load_train_checkpoint(cfg, model, optim)
    early_stopper = EarlyStopper(patience=15, min_delta=0)
    early_stop = False
    cudnn.benchmark = True
    
    logger.info("start epoch {}".format(start_epoch+1))
    #best_pred = 0
    isbest = False
    best_pred = 0
    best_policy = cu.best_policy(cfg) 
    data_type = cfg.DATA.DATA_TYPE
    if cfg.MODEL.MODEL_NAME == "Two_stream_fusion" and cfg.MODEL.FUSION_METHOD == "late":
        simple_loss = True
    else:
        simple_loss = False
    loss_pack_train = sol.loss_builder(loss_wtrain, data_type, simple_loss)
    loss_pack_test = sol.loss_builder(loss_wtest, data_type, simple_loss)
    
    for epoch in range(start_epoch+1, cfg.SOLVER.MAX_EPOCH):

        if torch.cuda.is_available():
            loss_dict = train_epoch(
                cfg, 
                model, 
                optim, 
                train_loader, 
                train_meter, 
                epoch, 
                data_type,
                loss_pack_train
            )
            if cfg.ENABLE_VAL:
                acc, loss = val_epoch(
                    cfg, 
                    model, 
                    test_loader, 
                    test_meter, 
                    epoch, 
                    data_type,
                    loss_pack_test,
                    loss_dict=loss_dict,
                )

                isbest = acc > best_pred
                if isbest:
                    best_pred = acc
                    isbest = best_policy.update(epoch, acc)  

                #implement early stopping here
                if early_stopper.early_stop(loss):
                     print("early stop in train_net is True")
                     early_stop = False #Change to True again when early stop is enabled
 
            trigger_save = cu.save_policy(epoch, isbest, cfg)
            if trigger_save:
                cu.save_checkpoint(
                    cfg.OUT_DIR,
                    cfg.CHECKPOINTS_FOLD, 
                    model, 
                    optim, 
                    epoch, 
                    cfg.NUM_GPUS,
                )   
        if early_stop:
            print("break!")
            break
    if cfg.ENABLE_VAL:
        logger.info("best model in {} epoch with acc {:.3f}".format(best_policy.best_epoch, best_policy.best_pred))
     
if __name__ == "__main__":
    train_net_only(cfg)
        