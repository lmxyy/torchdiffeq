import os

import foolbox
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as Data
import torchvision  # this is the database of torch
from torchvision import transforms

DOWNLOAD_MNIST = False  # whether you will download the mnist
use_gray = False  # 是否输出灰度图。如果选否则会输出差别更明显的热力图


def my_show_transform(x, use_gray):
    x = (x + 1.0) / 2.0
    """Imshow for Tensor."""
    x = np.clip(x, 0, 1)
    if use_gray:
        plt.imshow(x, cmap='gray')
    else:
        plt.imshow(x)
    return x


# instantiate the model
from fooling.Classifier import Classifier

CNN = Classifier()
CNN.load_state_dict(torch.load('c_params.pkl', map_location='cpu'))
CNN.eval()
if torch.cuda.is_available():
    CNN = CNN.cuda()
mean = np.array([0.5])
std = np.array([0.5])

fmodel = foolbox.models.PyTorchModel(
    CNN, bounds=(-1, 1), num_classes=10, preprocessing=(mean, std))

# get source image and label
my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    download=DOWNLOAD_MNIST,
    transform=my_transform
)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

tot = 0
correct = 0
wrong = 0

for batch_idx, (data, target) in enumerate(test_loader):
    if batch_idx >= 1000:
        break
    print(batch_idx)
    method_name = 'DeepFoolL2Attack'
    save_path = os.path.join(r'output/result_pic_mnist', method_name)
    tot += 1
    label = target.data.numpy()[0]
    print('label', label)
    image = data.squeeze(0).cpu().numpy()
    plt.subplot(1, 2, 1)
    my_show_transform(image.squeeze(), use_gray)
    predict = np.argmax(fmodel.predictions(image))
    print('predicted class', predict)
    if predict == label:
        correct += 1
    else:
        with open(os.path.join(save_path, 'log'), 'a') as f:
            f.write('%d: label: %d\tpredicted class: %d\n' % (batch_idx, label, predict))
        continue
    # 你需要改变这个method name，如果你用不同的方法
    makedirs(save_path)

    # 这里指定用什么方法去攻击。
    attack = foolbox.attacks.DeepFoolL2Attack()

    criterion = foolbox.criteria.Misclassification()
    adversarial = foolbox.Adversarial(fmodel, criterion, image, np.array(label, dtype=np.int64),
                                      distance=foolbox.distances.MSE)
    attack(adversarial)
    adversarial = adversarial.image

    adversarial_predict = np.argmax(fmodel.predictions(adversarial))
    print('adversarial class', adversarial_predict)
    if adversarial_predict != label:
        wrong += 1
    plt.subplot(1, 2, 2)

    if use_gray:
        plt.imshow(adversarial.squeeze(), cmap='gray')  #
    else:
        plt.imshow(adversarial.squeeze())  # , cmap ='gray'
    with open(os.path.join(save_path, 'log'), 'a') as f:
        f.write('%d: label: %d\tpredicted class: %d\tadversarial class: %d\n' % (
            batch_idx, label, predict, adversarial_predict))
    plt.savefig(os.path.join(save_path, '%d.png' % batch_idx))
print(tot, correct, wrong)
