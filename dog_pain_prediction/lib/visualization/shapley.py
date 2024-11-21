import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

class ShapleyExplainer:
    def __init__(self, model, ethogram_train, cfg):
            """
            Initialize the ShapleyExplainer with the model and training data.

            Args:
                model_class (type): The class of the model.
                checkpoint_path (str): Path to the checkpoint file.
                train_data (tuple): A tuple of training data tensors (X_train_frames, X_train_kp).
                ethogram_train (torch.Tensor): The training ethogram data tensor.
                cfg (dict): Configuration dictionary for the model.
            """
            self.model = model  # Initialize model with configuration
            self.model.eval()  # Set model to evaluation mode

            # self.train_frames, self.train_kp = train_data
            self.ethogram_train = ethogram_train

            # Initialize SHAP explainer with a subset of training data for efficiency
            self.explainer = shap.DeepExplainer(self.model.cuda(), self.ethogram_train.cuda())
    
    def predict(self, inputs):
        """
        Model prediction function including ethogram input.

        Args:
            inputs (tuple): A tuple of input tensors (frames, keypoints).

        Returns:
            np.ndarray: The model's predictions.
        """
        frames, kp = inputs
        ethogram_test = self.ethogram_test
        return self.model((ethogram_test).detach().numpy())
    
    def calculate_shap_values(self, ethogram_test):
        """
        Calculate SHAP values for the given test data.
        
        Args:
            test_data (tuple): A tuple of test data tensors (X_test_frames, X_test_kp).
            ethogram_test (torch.Tensor): The test ethogram data tensor.

        Returns:
            list: A list of SHAP values for each input tensor.
        """

        self.ethogram_test = ethogram_test  # Save for use in predict function
        print(f"this is the ethogram: {self.ethogram_test}")
        # self.ethogram_test.requires_grad = True
        # images = test_data[0]
        # images.requires_grad = True
        # kp = test_data[1]
        # kp.requires_grad = True
        return self.explainer.shap_values(ethogram_test)

    def summary_plot(self, shap_values, ethogram_test, save_path=None):
        # plt.figure()
        # shap.summary_plot(shap_values[0], test_data[0], feature_names=['Frame Features'], show=False)
        # if save_path:
        #     plt.savefig(save_path + '_frames.png')
        # plt.close()

        # plt.figure()
        # shap.summary_plot(shap_values[1], test_data[1], feature_names=['Keypoint Features'], show=False)
        # if save_path:
        #     plt.savefig(save_path + '_keypoints.png')
        # plt.close()
        idx_class_dict = self.get_idx_class()
        feature_names = [idx_class_dict[i] for i in range(ethogram_test.shape[1])]
        #ethogram_np = ethogram_test.numpy()
        plt.figure()
        shap.summary_plot(shap_values[:,:,0], ethogram_test, feature_names=feature_names, plot_type = 'bar', show=False)
        if save_path:
            plt.savefig(save_path + 'bar_ethogram.pdf')
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values[:,:,0], ethogram_test, feature_names=feature_names, show=False)
        if save_path:
            plt.savefig(save_path + 'directionality_ethogram.pdf')
        plt.close()

    def force_plot(self, all_shap_values, all_ethogram, all_labels, save_path=None):
        idx_class_dict = self.get_idx_class()
        feature_names = [idx_class_dict[i] for i in range(all_ethogram.shape[1])]
        class_index = self.explainer.expected_value[0]
        
        # Loop through all instances
        for instance_index in range(len(all_labels)):
            instance_ethogram = all_ethogram[instance_index]
            instance_shap_values = all_shap_values[instance_index, :, 0]
            label = all_labels[instance_index]

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap.Explanation(values=instance_shap_values, 
                                                base_values=class_index, 
                                                data=instance_ethogram, 
                                                feature_names=feature_names), 
                                show=False)
            plt.tight_layout()
            
            # Save the plot with a unique filename for each instance
            if save_path:
                filename = f"waterfall_{'pain' if label == 1 else 'not_pain'}_{instance_index}.pdf"
                plt.savefig(save_path + filename)
            plt.close()
    
    def force_plot2(self, all_shap_values, all_ethogram, all_labels, save_path=None):
        idx_class_dict = self.get_idx_class()
        feature_names = [idx_class_dict[i] for i in range(all_ethogram.shape[1])]
        class_index = self.explainer.expected_value[0]
        pain_instance_index = np.where(all_labels == 1)[0][0]
        pain_instance_ethogram = all_ethogram[pain_instance_index]
        pain_instance_shap_values = all_shap_values[pain_instance_index, :, 0]

        not_pain_instance_index = np.where(all_labels == 0)[0][0]
        not_pain_instance_ethogram = all_ethogram[not_pain_instance_index]
        not_pain_instance_shap_values = all_shap_values[not_pain_instance_index, :, 0]

        plt.figure(figsize=(10,6))
        shap.waterfall_plot(shap.Explanation(values=pain_instance_shap_values, 
                                            base_values=class_index, 
                                            data=pain_instance_ethogram, 
                                            feature_names=feature_names), 
                            show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + 'pain_waterfall.pdf')
        plt.close()

        # Plot waterfall plot for not-pain instance
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=not_pain_instance_shap_values, 
                                            base_values=class_index, 
                                            data=not_pain_instance_ethogram, 
                                            feature_names=feature_names), 
                            show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + 'not_pain_waterfall.pdf')
        plt.close()

    def get_idx_class(self):
        """
        originally 8: obscured, 9:UL, 6:sitting abnormally, but we deleted those earlier on"""
        dict = {
            0: 'standing still',
            1: 'walking',
            2: 'trotting',
            3: 'circling',
            4: 'jumping up',
            5: 'sitting',
            6: 'sitting abnormally',
            7: 'lying down',
            8: 'decreased weight bearing'
            }
        # dict = {
        #     0: 'standing still',
        #     1: 'walking',
        #     2: 'trotting',
        #     3: 'circling',
        #     4: 'jumping up',
        #     5: 'sitting',
        #     6: 'lying down',
        #     7: 'decreased weight bearing'
        #     }
        return dict

# Example usage
if __name__ == "__main__":
    # Assuming model, X_train_frames, X_train_kp, ethogram_train, X_test_frames, X_test_kp, ethogram_test are defined
    model = Two_stream_fusion(cfg)  # Your model initialization with appropriate configuration
    ethogram_train = ethogram_train

    explainer = ShapleyExplainer(model, ethogram_train)
    
    test_data = (X_test_frames, X_test_kp)
    ethogram_test = ethogram_test

    # Calculate SHAP values
    shap_values = explainer.calculate_shap_values(test_data, ethogram_test)
    
    # Generate summary plots
    explainer.summary_plot(shap_values, test_data, ethogram_test)
    
    # Generate a force plot for a single instance
    explainer.force_plot(shap_values, test_data, index=0)
