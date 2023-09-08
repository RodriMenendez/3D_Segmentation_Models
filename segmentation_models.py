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
        if self.model_architecture == 'unet':
            self.model = models.U_Net(self.spatial_dims, self.in_channels, self.out_channels, self.channels, self.strides, self.kernel_size)

        elif model_architecture == 'unetr':
            self.model = models.UNETR(self.in_channels, self.out_channels, self.image_shape)
        else:
            raise Exception(f'{self.model_architecture} model not supported')

    def load_weights(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        _ = self.model.double()
        _ = self.model.eval()

    def unsqueeze_image(self, image):
        if len(image.shape) >= self.spatial_dims + 2:
            return image.double()
        else:
            image = image.unsqueeze(0)
            return self.unsqueeze_image(image)

    def set_image(self, image):
        if isinstance(image, str):
            image = torch.tensor(imread(image))
        elif isinstance(image, torch.Tensor):
            pass
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image)
        else:
            raise Exception('image is not str, torch.Tensor, or numpy.ndarray')

        image = self.unsqueeze_image(image)
        self.image = image

    def predict(self):
        with torch.no_grad():
            if self.image is None:
                raise Exception('must set image first with self.set_image(image_path)')
            else:
                self.output = self.model(self.image)

        self.prediction = evaluations.prediction(self.output)

    def create_plots(self):
        if self.spatial_dims == 3:
            prediction_plot= np.argmax(self.prediction[0, 0], axis=0)
            image_plot = np.argmax(self.image[0, 0], axis=0)
        elif self.spatial_dims == 2:
            prediction_plot = self.prediction[0, 0]
            image_plot = self.image[0, 0]

        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(prediction_plot)
        axs[0].set_title('Prediction Max Intensity' if self.spatial_dims==3 else 'Prediction')
        axs[1].imshow(image_plot)
        axs[1].set_title('Input Max Intensity' if self.spatial_dims==3 else 'Input')
        plt.show()


    def __str__(self):
        return self.model.__str__()
