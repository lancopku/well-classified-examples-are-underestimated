from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from models import MNISTNet

# NOTE: This is a hack to get around "User-agent" limitations when downloading MNIST datasets
#       see, https://github.com/pytorch/vision/issues/3497 for more information
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

epsilons = [0, .05, .1, .15, .2,]
pretrained_model_ce = "/home/xxx/encourage_ns/lsoftmax-pytorch/bg0_le1.0_t0_m2_acc99.60_best.pth"
pretrained_model_el = "/home/xxx/encourage_ns/lsoftmax-pytorch/bg-1_le1.0_t0_m2_acc99.68_best.pth"
use_cuda=True


# MNIST Test dataset and dataloader declaration
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=1, shuffle=True,**kwargs)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model_ce = MNISTNet(margin=2, device=device).to(device)
model_el = MNISTNet(margin=2, device=device).to(device)

# Load the pretrained model
model_ce.load_state_dict(torch.load(pretrained_model_ce, map_location='cuda'))
model_el.load_state_dict(torch.load(pretrained_model_el, map_location='cuda'))
# Set the model in evaluation mode. In this case this is for the Dropout layers
model_ce.eval()
model_el.eval()
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        # output = F.log_softmax(output, dim=1)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # perturbed_data = data
        #
        # # Re-classify the perturbed image
        output = model(perturbed_data)
        output = F.log_softmax(output, dim=1)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct+= final_pred.eq(target.view_as(final_pred)).sum().item() #
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
    #
    # # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    # final_acc= 100. * float(correct) / len(test_loader.dataset) #
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples



# plot attack
accuracies = []
accuracies_ce=[]
accuracies_el=[]
# examples = []

# Run test for each epsilon
for eps in epsilons:
    acc_ce, _ = test(model_ce, device, test_loader, eps)
    acc_el, _ = test(model_el, device, test_loader, eps)
    accuracies_ce.append(acc_ce)
    accuracies_el.append(acc_el)
    # examples.append(ex)
plt.figure(figsize=(5,5))
plt.ylim(0.76,1)
plt.plot(epsilons, accuracies_ce, "r*-",label='Cross Entropy')
plt.plot(epsilons, accuracies_el, "b*-",label='Encouraging Loss')
plt.yticks(np.arange(0.76, 1.02, step=0.02))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon Based on Margin Method")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('attack_m2.png')
# plt.show()

# # Plot several examples of adversarial samples at each epsilon
# cnt = 0
# plt.figure(figsize=(8,10))
# for i in range(len(epsilons)):
#     for j in range(len(examples[i])):
#         cnt += 1
#         plt.subplot(len(epsilons),len(examples[0]),cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         if j == 0:
#             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
#         orig,adv,ex = examples[i][j]
#         plt.title("{} -> {}".format(orig, adv))
#         plt.imshow(ex, cmap="gray")
# plt.tight_layout()
# plt.show()