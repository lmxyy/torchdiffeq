import foolbox
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = np.transpose(inp, (1, 2, 0))  # .detach().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


# instantiate the model
resnet18 = models.resnet18(pretrained=True).eval()
if torch.cuda.is_available():
    resnet18 = resnet18.cuda()
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
fmodel = foolbox.models.PyTorchModel(
    resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))

# get source image and label
image, label = foolbox.utils.imagenet_example(data_format='channels_first')
image = image / 255.  # because our model expects values in [0, 1]
# print(image)
# imshow(image)
plt.imshow(np.transpose(image, (1, 2, 0)))
plt.show()

print('label', label)
print('predicted class', np.argmax(fmodel.predictions(image)))

# apply attack on source image
attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image, np.array(label, dtype=np.int64))

print('adversarial class', np.argmax(fmodel.predictions(adversarial)))
plt.imshow(np.transpose(adversarial, (1, 2, 0)))
plt.show()
