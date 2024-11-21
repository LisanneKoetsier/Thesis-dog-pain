import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import lib.data_build.data_utils as data_utils
from lib.model.two_stream import Two_stream_fusion

def load_model_from_checkpoint(checkpoint_path, cfg):
    # Instantiate the model
    model = Two_stream_fusion(cfg)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load the model state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def get_layer(model, layer_name):
    """
    Return the targeted layer (nn.Module Object) given a hierarchical layer name,
    separated by /.
    Args:
        model (model): model to get layers from.
        layer_name (str): name of the layer.
    Returns:
        prev_module (nn.Module): the layer from the model with `layer_name` name.
    """
    layer_ls = layer_name.split("/")
    prev_module = model
    for layer in layer_ls:
        prev_module = prev_module._modules[layer]

    return prev_module

class GradCAM:
    """
    GradCAM class helps create localization maps using the Grad-CAM method for input videos
    and overlap the maps over the input videos as heatmaps.
    https://arxiv.org/pdf/1610.02391.pdf
    """

    def __init__(
        self, model, target_layers, data_mean, data_std, colormap="viridis"
    ):
        """
        Args:
            model (model): the model to be used.
            target_layers (list of str(s)): name of convolutional layer to be used to get
                gradients and feature maps from for creating localization maps.
            data_mean (tensor or list): mean value to add to input videos.
            data_std (tensor or list): std to multiply for input videos.
            colormap (Optional[str]): matplotlib colormap used to create heatmap.
                See https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html
        """

        self.model = model
        # Run in eval mode.
        self.model.eval()
        self.target_layers = target_layers

        self.gradients = {}
        self.activations = {}
        self.colormap = plt.get_cmap(colormap)
        self.data_mean = data_mean
        self.data_std = data_std
        self._register_hooks()

    def _register_single_hook(self, layer_name):
        """
        Register forward and backward hook to a layer, given layer_name,
        to obtain gradients and activations.
        Args:
            layer_name (str): name of the layer.
        """

        def get_gradients(module, grad_input, grad_output):
            #print(f"grad output: {grad_output[0]}")
            self.gradients[layer_name] = grad_output[0].detach()

        def get_activations(module, input, output):
            # print(f"output activation: {output}")
            # print(f"output activation: {output[0][0].shape}")
            # #print(f"rist output activation: {output[0]}")
            # #print(f"second output activation: {output[1]}")
            # print(f"second output activation: {output[1].shape}")
            self.activations[layer_name] = output[0][0].clone().detach()

        target_layer = get_layer(self.model, layer_name=layer_name)
        print(f"layer_name: {layer_name}")
        print(f"target_layer: {target_layer}")
        target_layer.register_forward_hook(get_activations)
        target_layer.register_backward_hook(get_gradients)

    def _register_hooks(self):
        """
        Register hooks to layers in `self.target_layers`.
        """
        for layer_name in self.target_layers:
            self._register_single_hook(layer_name=layer_name)

    def _calculate_localization_map(self, inputs, labels=None):
        """
        Calculate localization map for all inputs with Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
        Returns:
            localization_maps (list of ndarray(s)): the localization map for
                each corresponding input.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        '''
        assert len(inputs) == len(
            self.target_layers
        ), "Must register the same number of target layers as the number of input pathways."
        '''
        # print(f"inputs: {inputs[0].shape}")
        # print(f"inputs length: {len(inputs)}")
        input_clone = [inp.clone() for inp in inputs]
        # print(f"input_clone: {input_clone}")
        # print(f"input_clone: {len(input_clone)}")
        preds = self.model(input_clone)
        preds.cuda()

        if labels is None:
            score = torch.max(preds, dim=-1)[0]
        else:
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            score = torch.gather(preds, dim=1, index=labels)

        self.model.zero_grad()
        score = torch.sum(score)
        score.backward()
        localization_maps = []
        for i, inp in enumerate(inputs[:1]):
            _, T, C, H, W = inp.size() # _, _, T, H, W = inp.size()

            #print(f"gradients: {self.gradients}")
            gradients = self.gradients[self.target_layers[i]]
            activations = self.activations[self.target_layers[i]]
            print(f"gradients size: {gradients.size()}")
            print(f"activations size: {activations.size()}")
            B, C, Tg, _, _ = gradients.size() #B, Tg, C, _, _ = gradients.size()

            weights = torch.mean(gradients.view(B, C, Tg, -1), dim=3)

            weights = weights.view(B, C, Tg, 1, 1)
            print(f"weights: {weights.shape}")
            #print(f"activations: {activations.shape}")
            localization_map = torch.sum(
                weights * activations, dim=1, keepdim=True
            )
            localization_map = F.relu(localization_map)
            localization_map = F.interpolate(
                localization_map,
                size=(T, H, W),
                mode="trilinear",
                align_corners=False,
            )
            localization_map_min, localization_map_max = (
                torch.min(localization_map.view(B, -1), dim=-1, keepdim=True)[
                    0
                ],
                torch.max(localization_map.view(B, -1), dim=-1, keepdim=True)[
                    0
                ],
            )
            localization_map_min = torch.reshape(
                localization_map_min, shape=(B, 1, 1, 1, 1)
            )
            localization_map_max = torch.reshape(
                localization_map_max, shape=(B, 1, 1, 1, 1)
            )
            # Normalize the localization map.
            localization_map = (localization_map - localization_map_min) / (
                localization_map_max - localization_map_min + 1e-6
            )
            localization_map = localization_map.data
            print(localization_map.size())
            localization_maps.append(localization_map)

        return localization_maps, preds

    def __call__(self, inputs, labels=None, alpha=0.5):
        """
        Visualize the localization maps on their corresponding inputs as heatmap,
        using Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
            alpha (float): transparency level of the heatmap, in the range [0, 1].
        Returns:
            result_ls (list of tensor(s)): the visualized inputs.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        result_ls = []
        localization_maps, preds = self._calculate_localization_map(
            inputs, labels=labels
        )
        for i, localization_map in enumerate(localization_maps):
            # Convert (B, 1, T, H, W) to (B, T, H, W)
            print(f"localization map: {localization_map.shape}")
            localization_map = localization_map.squeeze(dim=1)
            if localization_map.device != torch.device("cpu"):
                localization_map = localization_map.cpu()
            
            heatmap = self.colormap(localization_map)
            print(f"heatmap after colormap: {heatmap.shape}")
            heatmap = heatmap[:, :, :, :, :3]
            print(f"heatmap after operation: {heatmap.shape}")
            # Permute input from (B, C, T, H, W) to (B, T, H, W, C)
            # Own:  (B0, T1, C2, H3, W4) to (B, T, C, H, W)  -->  (0, 1, 3, 4, 2)
            # print(f"curr inp before: {inputs[i]}")
            # curr_inp = inputs[i].permute(0, 2, 3, 4, 1)
            curr_inp = inputs[i].permute(0, 1, 3, 4, 2)
            # print(f"curr inp after: {curr_inp}")
            if curr_inp.device != torch.device("cpu"):
                curr_inp = curr_inp.cpu()
            
            print(f"curr inp before tensor normalize: {curr_inp.shape}")
            curr_inp = data_utils.revert_tensor_normalize(
                curr_inp, self.data_mean, self.data_std
            )
            heatmap = torch.from_numpy(heatmap)
            print(f"heatmap: {heatmap.shape}")
            print(f"curr_inp: {curr_inp.shape}")

            curr_inp = alpha * heatmap + (1 - alpha) * curr_inp
            # Permute inp to (B, T, C, H, W)
            curr_inp = curr_inp.permute(0, 1, 4, 2, 3)
            result_ls.append(curr_inp)

        return result_ls, preds
    

if __name__ == "__main__":
    # Instantiate GradCAM
    model = load_model_from_checkpoint(cfg)
    target_layers = ['clstm.last_conv_layer', 'lstm_stream.last_conv_layer']  # Replace with actual layer names
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    gradcam = GradCAM(model=model, target_layers=target_layers, data_mean=data_mean, data_std=data_std)