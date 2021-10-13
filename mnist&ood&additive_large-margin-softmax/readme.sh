
### Installation
1.Clone the code from the Pytorch implementation of L-Softmax
2. Overwrite files

##CE
python3 train_mnist.py --gpu 0 --print_every_epoch 0 --print_every_run 1 --runs 5 --bonus_gamma -1 --log_end 1  --margin 1 --save 0
python3 train_mnist.py --gpu 1 --print_every_epoch 0 --print_every_run 1 --runs 5 --bonus_gamma 0 --log_end 1  --margin 1 --save 0

### CE with label smoothing
python3 train_mnist.py --gpu 0 --print_every_epoch 0 --print_every_run 1 --runs 5 --bonus_gamma -1 --log_end 1  --margin 1 --save 0 --label_smoothing 0.1
python3 train_mnist.py --gpu 1 --print_every_epoch 0 --print_every_run 1 --runs 5 --bonus_gamma 0 --log_end 1  --margin 1 --save 0 --label_smoothing 0.1

### additive with margin loss
python3 train_mnist.py --gpu 0 --print_every_epoch 0 --print_every_run 1 --runs 5 --bonus_gamma -1 --log_end 1  --margin 2 --save 0
python3 train_mnist.py --gpu 1 --print_every_epoch 0 --print_every_run 1 --runs 5 --bonus_gamma 0 --log_end 1  --margin 2 --save 0

### MSE
python3 train_mnist.py --gpu 0 --print_every_epoch 0 --print_every_run 1 --runs 5 --base_loss mse --bonus_gamma 0  --margin 1
#99.56 0.05
python3 train_mnist.py --gpu 1 --print_every_epoch 0 --print_every_run 1 --runs 5 --base_loss mse --bonus_gamma 2  --margin 1
#99.66 0.04



## attack with FGSM
python3 attack.py



## ood MNIST
gpu=0
name=baseline
python3 -m ood_test_mnist --model_path bg0_le1.0_acc99.41_best.pth \
--log_file ${name}.txt --output_name ${name} --gpu ${gpu} --ood fmnist --ind mnist

gpu=5
name=el
python3 -m ood_test_mnist --model_path bg-1_le1.0_acc99.58_best.pth \
--log_file ${name}.txt --output_name ${name} --gpu ${gpu} --ood fmnist --ind mnist


gpu=6
name=baseline_ls
python3 -m ood_test_mnist --model_path bg0_le1.0_m1_ls0.1_acc99.51_best.pth \
--log_file ${name}.txt --output_name ${name} --gpu ${gpu} --ood fmnist --ind mnist

gpu=7
name=el_ls
python3 -m ood_test_mnist --model_path bg-1_le1.0_m1_ls0.1_acc99.65_best.pth \
--log_file ${name}.txt --output_name ${name} --gpu ${gpu} --ood fmnist --ind mnist
