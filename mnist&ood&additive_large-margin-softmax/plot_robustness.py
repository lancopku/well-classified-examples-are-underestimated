# plot attack
import matplotlib.pyplot as plt
import numpy as np
accuracies = []
accuracies_ce=[]
accuracies_el=[]
# examples = []
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

# X、Y轴刻度标签字体大小
matplotlib.rcParams['axes.labelsize'] = 14

# epsilons = [0, .05, .1, .15, .2, .25, .3]
# accuracies_ce = [0.9933,0.9835,0.9628,0.9314,0.8875,0.8312,0.7744]
# accuracies_el = [0.9955,0.9903,0.9782,0.9588,0.9212,0.8577,0.776]
epsilons = [0, .05, .1, .15, .2,]
accuracies_ce = [0.9933,0.9835,0.9628,0.9314,0.8875]
accuracies_el = [0.9955,0.9903,0.9782,0.9588,0.9212]

# Epsilon: 0      Test Accuracy = 9933 / 10000 = 0.9933
# Epsilon: 0      Test Accuracy = 9955 / 10000 = 0.9955
# Epsilon: 0.05   Test Accuracy = 9835 / 10000 = 0.9835
# Epsilon: 0.05   Test Accuracy = 9903 / 10000 = 0.9903
# Epsilon: 0.1    Test Accuracy = 9628 / 10000 = 0.9628
# Epsilon: 0.1    Test Accuracy = 9782 / 10000 = 0.9782
# Epsilon: 0.15   Test Accuracy = 9314 / 10000 = 0.9314
# Epsilon: 0.15   Test Accuracy = 9588 / 10000 = 0.9588
# Epsilon: 0.2    Test Accuracy = 8875 / 10000 = 0.8875
# Epsilon: 0.2    Test Accuracy = 9212 / 10000 = 0.9212
# Epsilon: 0.25   Test Accuracy = 8312 / 10000 = 0.8312
# Epsilon: 0.25   Test Accuracy = 8577 / 10000 = 0.8577
# Epsilon: 0.3    Test Accuracy = 7744 / 10000 = 0.7744
# Epsilon: 0.3    Test Accuracy = 7760 / 10000 = 0.776


# Run test for each epsilon
# for eps in epsilons:
#     acc_ce, _ = test(model_ce, device, test_loader, eps)
#     acc_el, _ = test(model_el, device, test_loader, eps)
#     accuracies_ce.append(acc_ce)
#     accuracies_el.append(acc_el)
    # examples.append(ex)
fontsize=20
# plt.title('Penalise Incorrectness',)

plt.figure(figsize=(6.3,5))
plt.ylim(0.88, 1)
plt.plot(epsilons, accuracies_ce, "r*-",label='Cross-Entropy Loss')
plt.plot(epsilons, accuracies_el, "b*-",label='Encouraging Loss')
plt.xlim(-0.001,0.201)
plt.yticks(np.arange(0.88, 1.002, step=0.02))
plt.xticks(np.arange(0, .25, step=0.05))
plt.title("Accuracy vs Epsilon (Perturbation Amount)",fontdict={'fontsize': 18,})
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.legend(fontsize=18)
# plt.show()
plt.savefig('attack.pdf')
# plt.savefig('attack.eps')



# Epsilon: 0      Test Accuracy = 9956 / 10000 = 0.9956
# Epsilon: 0      Test Accuracy = 9967 / 10000 = 0.9967
# Epsilon: 0.05   Test Accuracy = 9898 / 10000 = 0.9898
# Epsilon: 0.05   Test Accuracy = 9903 / 10000 = 0.9903
# Epsilon: 0.1    Test Accuracy = 9770 / 10000 = 0.977
# Epsilon: 0.1    Test Accuracy = 9780 / 10000 = 0.978
# Epsilon: 0.15   Test Accuracy = 9586 / 10000 = 0.9586
# Epsilon: 0.15   Test Accuracy = 9629 / 10000 = 0.9629
# Epsilon: 0.2    Test Accuracy = 9289 / 10000 = 0.9289
# Epsilon: 0.2    Test Accuracy = 9381 / 10000 = 0.9381
# Epsilon: 0.25   Test Accuracy = 8836 / 10000 = 0.8836
# Epsilon: 0.25   Test Accuracy = 9057 / 10000 = 0.9057
# Epsilon: 0.3    Test Accuracy = 8268 / 10000 = 0.8268
# Epsilon: 0.3    Test Accuracy = 8602 / 10000 = 0.8602