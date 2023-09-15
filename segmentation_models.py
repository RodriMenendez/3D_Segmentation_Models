from benchmarking import models, evaluations, transformations
import torch
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import glob


class Model():
    def __init__(self, model_architecture, spatial_dims, in_channels, out_channels, channels, strides, kernel_size, image_shape=None):
        self.image = None
        self.output = None
        self.prediction = None
        self.mask = None
        self.model = None
        self.model_architecture = model_architecture.lower()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.image_shape = image_shape

        self.load_model()


    def load_model(self):
        """
        Builds a model with U-Net or UNETR architecture
        """
        if self.model_architecture == 'unet':
            self.model = models.U_Net(self.spatial_dims, self.in_channels, self.out_channels, self.channels, self.strides, self.kernel_size)

        elif self.model_architecture == 'unetr':
            self.model = models.UNETR(self.in_channels, self.out_channels, self.image_shape)
        else:
            raise Exception(f'{self.model_architecture} model not supported')

    def load_weights(self, model_path):
        """
        Loads weights from a state dictionary

        model_path: filepath of torch state dictionary to be loaded
        """
        self.model.load_state_dict(torch.load(model_path))
        _ = self.model.double()
        _ = self.model.eval()

    def unsqueeze_image(self, image):
        if len(image.shape) >= self.spatial_dims + 2:
            return image.double()
        else:
            image = image.unsqueeze(0)
            return self.unsqueeze_image(image)
    
    def get_image(self, image):
        if isinstance(image, str):
            image = torch.tensor(imread(image))
        elif isinstance(image, torch.Tensor):
            pass
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image)
        else:
            raise Exception('image is not str, torch.Tensor, or numpy.ndarray')

        image = self.unsqueeze_image(image)
        return image

    def set_image(self, image):
        """
        Sets the image(s) to be used for prediction. 

        image: can be of type str (filepath of image), torch.Tensor, or numpy.ndarray
        """
        self.image = self.get_image(image)

    def set_mask(self, mask):
        """
        Sets the mask(s) to be used for metric calculation

        mask: can be of type str (filepath of mask), torch.Tensor, or numpy.ndarray
        """
        self.mask = self.get_image(mask)

    def predict(self):
        """
        Model segmentation of currently loaded image(s)
        """
        with torch.no_grad():
            if self.image is None:
                raise Exception('must set image first with self.set_image(image_path)')
            else:
                self.output = self.model(self.image)

        self.prediction = evaluations.prediction(self.output)

    def save_prediction(self, path):
        """
        Save currently loaded prediction
        """
        torch.save(self.prediction, path)

    def create_plots(self, cmap='viridis', alpha_range=0.9):
        """
        Plots depth index with max intensity for 3D, regular plots for 2D

        cmap: str or matplotlib cmap to be used for plots
        """
        def normalize_alpha(alpha, alpha_range):
            alpha = alpha/alpha.max()
            alpha = alpha*alpha_range + (1-alpha_range)
            return alpha

        if self.prediction.shape[0] > 1:
            for prediction in self.prediction:
                self.create_plots(prediction.unsqueeze(0))
        else:
            if self.spatial_dims == 3:
                prediction_alpha, prediction_plot= torch.max(self.prediction[0, 0], axis=0)
                image_alpha, image_plot = torch.max(self.image[0, 0], axis=0)
                image_alpha = normalize_alpha(image_alpha, alpha_range)
                mask_alpha, mask_plot = torch.max(self.mask[0, 0], axis=0)
                mask_alpha = normalize_alpha(mask_alpha, alpha_range)
            elif self.spatial_dims == 2:
                prediction_plot = self.prediction[0, 0]
                image_plot = self.image[0, 0]
                mask_plot = self.mask[0, 0]

            fig, axs = plt.subplots(1, 3, figsize=(10, 10))
            if self.spatial_dims == 3:
                title = ' Max Intensity'
            else:
                title = ''
                prediction_alpha = None
                image_alpha = None
                mask_alpha = None

            axs[0].imshow(prediction_plot, cmap=cmap, alpha=prediction_alpha)
            axs[0].set_title('Prediction'+title)
            axs[1].imshow(image_plot, cmap=cmap, alpha=image_alpha)
            axs[1].set_title('Input'+title)
            axs[2].imshow(mask_plot, cmap=cmap, alpha=mask_alpha)
            axs[2].set_title('Mask'+title)
            plt.show()

    def get_evals(self):
        """
        Calculates the confusion matrix, IoU, precision, and accuracy of the current prediction based on the current mask

        Returns
        evals: dictionary {confusion matrix, IoU, precision, accuracy}
        """
        confusion_matrix = evaluations.ConfusionMatrix(self.prediction, self.mask)
        iou = evaluations.IoU(confusion_matrix)
        precision = evaluations.precision(confusion_matrix)
        accuracy = evaluations.accuracy(confusion_matrix)

        evals = {'confusion matrix': confusion_matrix, 'iou': iou, 'precision': precision, 'accuracy': accuracy}

        return evals


    def __str__(self):
        return self.model.__str__()
