import numpy as np
import pprint
from tqdm import tqdm
import os
import cv2
import copy
import joblib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import RobustScaler
from PIL import Image
from treeinterpreter import treeinterpreter as ti

# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from lib.data_build.data_utils import loss_weight
import lib.model.two_stream as model_set
from lib.data_build import Data_loader
import lib.utils.logging as log
import lib.utils.checkpoint as cu
from lib.utils.meter import Test_meter
import lib.utils.solver as sol
from lib.visualization import GradCAM

logger = log.get_logger(__name__)


def build_model(cfg, decision_tree):

    if cfg.MODEL.TYPE == "two_stream" or cfg.MODEL.TYPE == "ethogram":
        model = eval(f"model_set.{cfg.MODEL.MODEL_NAME}(cfg, decision_tree).cuda()")
    else:
        model = eval(f"model_set.{cfg.MODEL.MODEL_NAME}(cfg).cuda()")

    test_meter = Test_meter(cfg)
    return (
        model,
        test_meter,
    )


def test_net(cfg):
    log.setup_logging(cfg.OUT_DIR, cfg.CHECKPOINTS_FOLD)
    logger.info(pprint.pformat(cfg))

    if cfg.MODEL.DECISION_TREE:
        decision_tree_path = os.path.join(cfg.CHECKPOINTS_FOLD, 'random_forest_model.joblib')
        rf_classifier = joblib.load(decision_tree_path)
    else:
        rf_classifier = None

    (
        model,
        test_meter,
    ) = build_model(cfg, rf_classifier)

    data_container = Data_loader(cfg)
    logger.info("start load dataset")

    test_loader, loss_weight = data_container.construct_loader("test")
    #labels = (np.array(test_loader.dataset.label_list)[:, -1]).astype(int)
    test_epoch = cu.load_test_checkpoint(cfg, model)
    cudnn.benchmark = True

    logger.info("Testing model for {} iterations".format(len(test_loader)))

    val_epoch(cfg, model, test_loader, test_meter, rf_classifier, test_epoch)

    if cfg.MODEL.TYPE == "rgb":
        grad_cam_epoch(cfg, model, test_loader)
    
def decision_tree_epoch(cfg, rf_classifier, ethograms, labels, start_name):
    idx_to_behavior_class = {
        0: 'standing still',
        1: 'walking',
        2: 'trotting',
        3: 'circling',
        4: 'jumping up',
        5: 'sitting',
        6: 'sitting abnormally',
        7: 'lying down',
        8: 'obscured',
        9: 'UL',
        10: 'decreased weight bearing'
        }
    idx_to_pain_class = {
        0: 'no pain',
        1: 'pain'
    }
    
    
    ethogram_data_boolean = (ethograms > 0)
    # Convert all non-zero values to True (boolean array)
    ethogram_data_boolean = (ethogram_data_boolean > 0)
    ethogram_data_boolean_np = ethogram_data_boolean.numpy()

    # Select a specific instance to explain
    instance_index = 0  # Index of the specific instance you want to explain
    instance = ethogram_data_boolean_np[instance_index].reshape(1, -1)
    labels_np = labels.cpu().numpy()
    label = labels_np[instance_index].reshape(1,-1)[0]

    # Get feature contributions using 
    print(f"this is the ethogram data boolean: {ethogram_data_boolean}")
    print(f"this is instance: {instance}")
    print(f"this is the labels: {labels}")
    print(f"this is the label: {label}")
    prediction, bias, contributions = ti.predict(rf_classifier, instance)

    # Print shapes to understand the structure
    print("Bias shape:", bias)  # Typically (n_classes,)
    print("Prediction shape:", prediction.shape)  # Typically (1, n_classes)
    print("Contributions shape:", contributions.shape)  # Typically (1, n_features, n_classes)

    # Define feature names for the ethogram dimensions
    feature_names = [idx_to_behavior_class[i] for i in range(ethogram_data_boolean.shape[1])]
    # Adjust feature names to exclude the last two and include the last one
    adjusted_feature_names = feature_names[:-3] + [feature_names[-1]]

    print(f"the feature names: {adjusted_feature_names}")
    # Get the index of the class with the highest predicted probability
    class_index = np.argmax(prediction[0])

    # Extract contributions for the specified class
    contributions_class = contributions[0][:-3] + contributions[0][-1:]
    contributions_class = contributions_class[:, class_index]

    # Print out the contributions
    print("Bias (base value):", bias[0][class_index])
    print("Prediction:", prediction[0][class_index])
    for name, contribution in zip(adjusted_feature_names, contributions_class):
        print(f"{name}: {contribution}")

    # Sort the contributions for better visualization
    sorted_indices = np.argsort(contributions_class)
    #sorted_indices = sorted_indices[:-2]

    print(f"this is the label: {label}")

    # Create a horizontal bar plot of the feature contributions
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_indices)), contributions_class[sorted_indices], align='center')
    plt.yticks(range(len(sorted_indices)), np.array(feature_names)[sorted_indices])
    plt.xlabel(f'Contribution to Prediction: {idx_to_pain_class[class_index]}')
    plt.ylabel('Features')
    plt.title(f'Feature Contributions for Specific Prediction for {start_name[instance_index]} with ground truth {idx_to_pain_class[label[0]]}')

    # Save the plot to a directory
    output_dir = "/home/albert/Lisanne/data/own_data/out/decision_tree_contributions"
    output_path = os.path.join(output_dir, f'feature_contributions {start_name[instance_index]}.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    

def grad_cam_epoch(cfg, model, test_loader):
    model.eval()
    print(f"this is the model: {model}")
    for cur_iter, (inputs, labels, ethogram, start_name) in enumerate(tqdm(test_loader)):

        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        outputs = model(inputs)

        preds = torch.argmax(outputs, dim=1)

        target_layers = ['clstm/convlstm']

        #target_layers = [model.clstm.convlstm]   #cell_list/1', 'clstm/convlstm/cell_list/2', 'clstm/convlstm/cell_list/2']
        
        grad_cam = GradCAM(model=model, target_layers=target_layers, data_mean=cfg.DATA.MEAN, data_std=cfg.DATA.STD)
        # print(f"inputs test: {inputs}")
        # print(f"inputs shape test: {len(inputs)}")
        vis, preds = grad_cam(inputs, labels=None, alpha=0.5)

        for idx, result in enumerate(vis):
            save_dir = r"/home/albert/Lisanne/data/own_data/out/gradcam"
            for t in range(result.shape[1]):
                frame = result[0, t].permute(1, 2, 0).numpy()
                frame = (frame * 255).astype(np.uint8)
                img = Image.fromarray(frame)
                img.save(os.path.join(save_dir, f'image_{idx}_frame_{start_name[idx]}.png'))

        # grad_cam = GradCAM(model=model, target_layers=target_layers)

        # print(f"these are the image shape: {inputs[0].shape}")

        # for input_tensor in inputs:
        #     # for j in range(inputs.shape[1]):
        #     #     print(f"one iteration yes!")
        #         #input_tensor = inputs[i,j].unsqueeze(0).unsqueeze(1)
        #         target_label = preds[i].item()
        #         targets = [ClassifierOutputTarget(target_label)]

        #         print(f"input tensor shape: {input_tensor.size()}")
        #         grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)
        #         grayscale_cam = grayscale_cam[0:, :] #just show for one images in the batch

        #         # Prepare the image for visualization
        #         img = inputs[i,j].cpu().numpy().transpose(1, 2, 0)
        #         img = np.clip(img, 0, 1)  # Ensure the pixel values are between 0 and 1

        #         # Convert grayscale CAM to heatmap
        #         cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        #         # Save the image with the heatmap
        #         output_path = os.path.join(cfg.OUT_DIR, 'gradcam', f'gradcam_batch{cur_iter}_{start_name}.jpg')
        #         cv2.imwrite(output_path, cam_image)

        #         print(f'Saved Grad-CAM image to {output_path}')



@torch.no_grad()
def val_epoch(cfg, model, test_loader, test_meter, rf_classifier, epoch=-1):
    model.eval()
    test_meter.time_start()
    for cur_iter, (inputs, labels, ethogram, start_name) in enumerate(tqdm(test_loader)):
        test_meter.time_pause()
        test_meter.update_data()
        test_meter.time_start()

        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        labels = labels.cuda()
        labels = labels.unsqueeze(-1)
        if cfg.MODEL.TYPE == "ethogram":
            ethogram_normalized = copy.deepcopy(ethogram)
            ethogram_normalized[ethogram_normalized>1] = 1 #todo uncomment
            print(f"this is ethogram normalized for test: {ethogram_normalized}")
            if rf_classifier is not None:
                print(f"this is rf classifier: {rf_classifier}")
                decision_tree_epoch(cfg, rf_classifier, ethogram_normalized, labels, start_name)
            outputs = model(inputs, ethogram_normalized.cuda())
        elif cfg.MODEL.TYPE == "rgb":
            outputs = model(inputs)
        else:
            outputs = model(inputs, [])


        if cfg.MODEL.MODEL_NAME == "Two_stream_fusion" and cfg.MODEL.FUSION_METHOD == "late":
            loss_func = nn.BCELoss().cuda()
        else:
            loss_func = nn.BCEWithLogitsLoss().cuda()
        loss = loss_func(outputs, labels.float())

        if cfg.MODEL.NUM_LABELS == 2:
            acc = sol.get_binary_acc(outputs, labels)
        else:
            acc = sol.get_accuracy(outputs, labels)

        loss, acc = (
            loss.item(),
            acc.item(),
        )
        batch_size = inputs[0].size(0)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.cpu()
        labels  = labels.cpu()
        if cfg.MODEL.TYPE == "ethogram":
            ethogram = ethogram.cpu()
        
        test_meter.update_states(loss, acc, batch_size, outputs, labels, start_name, ethogram)

        test_meter.time_pause()
        test_meter.update_batch()
        test_meter.time_start()

    test_meter.update_epoch(epoch)
    acc = test_meter.acc_meter.avg
    f1_score = test_meter.f1
    test_meter.reset()
    return acc, f1_score




# test_full is never used?
def test_full(cfg):
    log.setup_logging(cfg.OUT_DIR, cfg.CHECKPOINTS_FOLD)
    logger.info(pprint.pformat(cfg))
    checkpoints_set = cu.get_checkpoints_set(cfg)
    assert len(checkpoints_set) > 0, f"no checkpoints file avalible in {cfg.CHECKPOINTS_FOLD}"
    
    (
        model,
        test_meter,
    ) = build_model(cfg)

    data_container = Data_loader(cfg)
    logger.info("start load dataset")

    test_loader = data_container.construct_loader("test")
    cudnn.benchmark = True
  
    best_preds = 0
    best_f1 = 0
    for file in checkpoints_set:
        checkpoint = torch.load(file)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['model_state'])
      
        acc, f1_score = val_epoch(cfg, model, test_loader, test_meter, epoch)
        if acc > best_preds:
            best_preds, best_f1, best_epoch = acc, f1_score, epoch
    logger.info("best model in {} epoch with acc {:.3f}, f1 score {:.3f}".format(
        best_epoch, best_preds, best_f1))