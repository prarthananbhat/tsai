from torch.nn import functional as F
import torch
import matplotlib.pyplot as plt
import cv2
from utils import denormalize
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class GradCAM:
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers 
    target_layers = list of convolution layer index as shown in summary
    """
    def __init__(self, model, candidate_layers=None):
        def save_fmaps(key):
          def forward_hook(module, input, output):
              self.fmap_pool[key] = output.detach()

          return forward_hook

        def save_grads(key):
          def backward_hook(module, grad_in, grad_out):
              self.grad_pool[key] = grad_out[0].detach()

          return backward_hook

        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.nll).to(self.device)
        print(one_hot.shape)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:] # HxW
        self.nll = self.model(image)
        #self.probs = F.softmax(self.logits, dim=1)
        return self.nll.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.nll.backward(gradient=one_hot, retain_graph=True)

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        # need to capture image size duign forward pass
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        # scale output between 0,1
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

def gradcam(images, labels, model, target_layers):
      model.eval()
      # map input to device
      images = images.to(device)
      # set up grad cam
      gcam = GradCAM(model, target_layers)
      # forward pass
      probs, ids = gcam.forward(images)
      # outputs agaist which to compute gradients
      ids_ = labels.view(len(images),-1).to(device)
      # backward pass
      gcam.backward(ids=ids_)
      layers = []
      for i in range(len(target_layers)):
            target_layer = target_layers[i]
            print("Generating Grad-CAM @{}".format(target_layer))
            # Grad-CAM
            layers.append(gcam.generate(target_layer=target_layer))
      # remove hooks when done
      gcam.remove_hook()
      return layers, probs, ids


def plot_gradcam_1(gcam_layers, images, target_classes, target_layers, classes, predicted_classes,channel_means,channel_stdevs):
    i_index = len(images)
    l_index = len(target_layers)
    for j in range(0, i_index):
        image_actual_class = classes[target_classes[j]]
        image_predicted_class = classes[predicted_classes[j][0]]
        fig = plt.figure(figsize=(12, 8))
        plt.subplot(1, l_index + 1, 1)
        plt_image = denormalize(images[j].cpu(), channel_means, channel_stdevs)
        img = np.transpose(plt_image, (1, 2, 0))
        img = np.uint8(255 * img)
        plt.tight_layout()
        plt.axis('off')
        plt.title("Actual %s \n Predicted %s" % (image_actual_class, image_predicted_class))
        plt.imshow(img, interpolation='bilinear')
        for i in range(0, l_index):
            plt.subplot(1, l_index + 1, i + 2)
            heatmap = 1 - gcam_layers[i][j].cpu().numpy()[0]  # reverse the color map
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.resize(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), (128, 128))
            plt.tight_layout()
            plt.title(target_layers[i])
            plt.axis('off')
            plt.imshow(superimposed_img, interpolation='bilinear')
