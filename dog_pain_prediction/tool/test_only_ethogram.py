import numpy as np
import pprint
from tqdm import tqdm
import os
import copy
import torch
import joblib
import torch.nn as nn
import torch.backends.cudnn as cudnn

from lib.data_build.data_utils import loss_weight
import lib.model.two_stream as model_set
from lib.data_build import Data_loader
import lib.utils.logging as log
import lib.utils.checkpoint as cu
from lib.utils.meter import Test_meter
import lib.utils.solver as sol

import matplotlib.pyplot as plt
from treeinterpreter import treeinterpreter as ti

from lib.visualization.shapley import ShapleyExplainer

logger = log.get_logger(__name__)


def build_model(cfg, decision_tree):

    if cfg.MODEL.TYPE == "two_stream" or cfg.MODEL.TYPE == "ethogram" or cfg.MODEL.TYPE == "ethogram_only":
        model = eval(f"model_set.{cfg.MODEL.MODEL_NAME}(cfg, decision_tree).cuda()")
    else:
        model = eval(f"model_set.{cfg.MODEL.MODEL_NAME}(cfg).cuda()")

    test_meter = Test_meter(cfg)
    return (
        model,
        test_meter,
    )


def test_net_only(cfg):
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
    train_loader, loss_weight_train = data_container.construct_loader("train")
    
    for batch in train_loader:
        _, _, train_ethogram, _ = batch
        # train_frames, train_keypoints = inputs
        break
    
    # print(f"this is the train frames: {train_frames}")
    # print(f"this is the train keypoints: {train_keypoints}")
    # print(f"this is the train ethogram: {train_ethogram}")
    test_epoch = cu.load_test_checkpoint(cfg, model)
    cudnn.benchmark = True

    logger.info("Testing model for {} iterations".format(len(test_loader)))
    shapley_explainer = None

    if cfg.MODEL.TYPE == "ethogram_only" and cfg.MODEL.DECISION_TREE == False:
        print(f"linear version of ethogram")
        print(f"this is cfg: {cfg}")
        shapley_explainer = ShapleyExplainer(model, train_ethogram, cfg) #add optional version
    val_epoch(cfg, model, test_loader, test_meter, test_epoch, shapley = shapley_explainer, rf_classifier=rf_classifier)

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
    for instance_index in range(len(ethogram_data_boolean_np)):
    #instance_index = 0  # Index of the specific instance you want to explain
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
        contributions_first_part = contributions[0][:8]  # First 8 features
        contributions_last_part = contributions[0][-1:]  # Last feature
        print(f"contributions first: {contributions_first_part}")
        print(f"contributions second: {contributions_last_part}")
        contributions_class = np.concatenate((contributions_first_part, contributions_last_part), axis=0)
        contributions_class = contributions_class[:, class_index]
        print(f"len features: {len(adjusted_feature_names)}")
        print(f"contributions: {contributions[0]}")
        print(f"contributions class: {contributions_class}")
        print(f"len contributions: {len(contributions_class)}")

        # Prepare feature names with raw values for plotting
        feature_values = instance.flatten()[:8].tolist() + [instance.flatten()[-1]]
        feature_names_with_values = [f"{name} ({value})" for name, value in zip(adjusted_feature_names, feature_values)]

        # Print out the contributions
        print("Bias (base value):", bias[0][class_index])
        print("Prediction:", prediction[0][class_index])
        for name, contribution in zip(adjusted_feature_names, contributions_class):
            print(f"{name}: {contribution}")

        # Sort the contributions for better visualization
        sorted_indices = np.argsort(contributions_class)
        #sorted_indices = sorted_indices[:-2]

        print(f"this is the label: {label}")
        title_name = start_name[instance_index].split("_")[0]
        # Create a horizontal bar plot of the feature contributions
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(contributions_class)), contributions_class, align='center')
        plt.yticks(range(len(contributions_class)), feature_names_with_values, fontsize=12)
        plt.xlabel(f'Contribution to Prediction: {idx_to_pain_class[class_index]}', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.title(f'Feature Contributions for {title_name} with ground truth {idx_to_pain_class[label[0]]}', fontsize=14)

        # Save the plot to a directory
        output_dir = "/home/albert/Lisanne/data/own_data/out/decision_tree_contributions"
        output_path = os.path.join(output_dir, f'feature_contributions {start_name[instance_index]}.png')
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    


#@torch.no_grad()
def val_epoch(cfg, model, test_loader, test_meter, epoch=-1, shapley=None, rf_classifier=None):
    model.eval()
    test_meter.time_start()
    ethogram_list = []
    shap_values_list = []
    labels_list = []
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
            ethogram_normalized[ethogram_normalized>1] = 1
            #print(f"this is the ethogram normalized: {ethogram_normalized}")
            if rf_classifier is not None:
                decision_tree_epoch(cfg, rf_classifier, ethogram_normalized, labels, start_name)
            outputs = model(inputs, ethogram_normalized.cuda()) #reminder changed to normal ethogram to test
        elif cfg.MODEL.TYPE == "ethogram_only":
            ethogram_normalized = copy.deepcopy(ethogram)
            ethogram_normalized[ethogram_normalized>1] = 1
            #print(f"this is the ethogram normalized: {ethogram_normalized}")
            # print(f"this is the train normalized ethogram: {ethogram_normalized}")
            # print(f"this is the train og ethogram: {ethogram}")
            if rf_classifier is not None:
                decision_tree_epoch(cfg, rf_classifier, ethogram_normalized, labels, start_name)
            outputs = model(ethogram_normalized.cuda()) #REMINDER: changed to normal ethogram to test
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
        if cfg.MODEL.TYPE == "ethogram_only":
            #print(f"ethogram: {ethogram.shape[0]}")
            batch_size = ethogram.shape[0]
            #print(f"batch size: {batch_size}")
        else:
            batch_size = inputs[0].size(0)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.cpu()
        labels  = labels.cpu()
        if cfg.MODEL.TYPE == "ethogram" or cfg.MODEL.TYPE == "ethogram_only":
            ethogram = ethogram.cpu()
        else:
            ethogram = []
        
        test_meter.update_states(loss, acc, batch_size, outputs, labels, start_name, ethogram)

        if shapley:
            # Calculate SHAP values for the current batch
            ethogram_normalized = copy.deepcopy(ethogram)
            ethogram_normalized[ethogram_normalized>1] = 1
            shap_values = shapley.calculate_shap_values(ethogram_normalized)
            shap_values_list.append(shap_values)
            ethogram_list.append(ethogram_normalized)
            labels_list.append(labels)
            # Save the SHAP summary plot for the first batch as an example
            # if cur_iter == 0:
            #     print(f"ethogram: {ethogram.shape}")
            #     print(f"ethogram type: {type(ethogram)}")
            #     print(f"shap values: {shap_values.shape}")
            #     print(f"type shape values: {type(shap_values)}")
            #     feature_names = [f'Feature {i}' for i in range(ethogram.shape[1])]
            #     print(f"feature names: {feature_names}")
            #     save_path = f"{cfg.OUT_DIR}/shap_summary_epoch{epoch}_batch{cur_iter}"
            #     shapley.summary_plot(shap_values, ethogram, save_path)

        test_meter.time_pause()
        test_meter.update_batch()
        test_meter.time_start()

    if shapley:
        # Concatenate collected data
        all_ethogram = torch.cat(ethogram_list, dim=0).numpy()
        all_shap_values = np.concatenate(shap_values_list, axis=0)
        all_labels = torch.cat(labels_list, dim=0).numpy()

        # Indices to exclude
        exclude_indices = [8, 9] #obscured and UL

        # Filter out indices 8 and 9
        all_ethogram = np.delete(all_ethogram, exclude_indices, axis=1)
        all_shap_values = np.delete(all_shap_values, exclude_indices, axis=1)

        print(f"all ethogram: {all_ethogram[0]}")
        print(f"all shap values: {shap_values}")

        # Plot summary plots
        summary_save_path = f"{cfg.OUT_DIR}/shap_summary_epoch{epoch}_test"
        force_save_path = f"{cfg.OUT_DIR}/shap_force_epoch{epoch}_test"
        shapley.summary_plot(all_shap_values, all_ethogram, summary_save_path)
        shapley.force_plot(all_shap_values, all_ethogram, all_labels, force_save_path)
        
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