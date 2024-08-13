import torchvision
import warnings
warnings.simplefilter("ignore")

# compute nodes do not have internet so download the data in advance

_ = torchvision.datasets.MNIST(root='data',
                               train=True,
                               transform=None,
                               target_transform=None,
                               download=True)
