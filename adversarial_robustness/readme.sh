parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--log_end', type=float, default=1,
                    help='')
parser.add_argument('--gpu', type=int, default=4,
                    help='-1 means cpu')
parser.add_argument('--loss', type=int, default=0,
                    help='0 ce 1 el')
parser.add_argument('--mode', type=str, default='train',choices=['train','adv_train','adv_test','test'],
                    help='')
parser.add_argument('--load', type=str, default='xxx.pth',)
parser.add_argument('--output_name', type=str, default='mnist',
                    help='')
parser.add_argument('--test_atk', type=str, default='pgd',choices=['fgsm','pgdl2','pgd','cw'],
                    help='')
parser.add_argument('--train_atk', type=str, default='pgd',
                    help='')
parser.add_argument('--tae', type=float, default=0.3,
                    help='')
args = parser.parse_args()

# lanco13

#1.  train
cd /home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos
screen -r a7

python3 attack.py --mode train --loss 0 --gpu 0 --output_name mnist_ce
python3 attack.py --mode train --loss 1 --gpu 1 --output_name mnist_el

#2.  adv train
tae=3
for train_atk in pgdl2
do
python3 attack.py --mode adv_train --loss 0 --gpu 2 --output_name mnist_ce_train_${train_atk}_${tae} \
--train_atk ${train_atk} --tae ${tae}
done

tae=3
for train_atk in pgdl2
do
python3 attack.py --mode adv_train --loss 1 --gpu 3 --output_name mnist_el_train_${train_atk}_${tae} \
 --train_atk ${train_atk} --tae ${tae}
done


tae=1
for train_atk in cw
do
python3 attack.py --mode adv_train --loss 0 --gpu 0 --output_name mnist_ce_train_${train_atk}_${tae} \
--train_atk ${train_atk} --tae ${tae}
done

tae=1
for train_atk in cw
do
python3 attack.py --mode adv_train --loss 1 --gpu 1 --output_name mnist_el_train_${train_atk}_${tae} \
 --train_atk ${train_atk} --tae ${tae}
done



#pgd fgsm pgdl2









fgsm fgsm
eps 0.3
Standard accuracy, ei 0.000000: 95.43 %
Standard accuracy, ei 0.050000: 92.59 %
Standard accuracy, ei 0.100000: 91.01 %
Standard accuracy, ei 0.150000: 89.79 %
Standard accuracy, ei 0.200000: 89.30 %
Standard accuracy, ei 0.250000: 89.09 %
Standard accuracy, ei 0.300000: 87.87 %

Standard accuracy, ei 0.000000: 97.21 %
Standard accuracy, ei 0.050000: 95.81 %
Standard accuracy, ei 0.100000: 95.18 %
Standard accuracy, ei 0.150000: 95.16 %
Standard accuracy, ei 0.200000: 95.33 %
Standard accuracy, ei 0.250000: 95.09 %
Standard accuracy, ei 0.300000: 94.15 %

#baseline
#eps 0.2
#Standard accuracy, ei 0.000000: 98.50 %
#Standard accuracy, ei 0.050000: 97.79 %
#Standard accuracy, ei 0.100000: 96.66 %
#Standard accuracy, ei 0.150000: 95.38 %
#Standard accuracy, ei 0.200000: 93.72 %
#Standard accuracy, ei 0.250000: 91.88 %
#Standard accuracy, ei 0.300000: 88.70 %
#
#el
#Standard accuracy, ei 0.000000: 98.54 %
#Standard accuracy, ei 0.050000: 97.78 %
#Standard accuracy, ei 0.100000: 96.87 %
#Standard accuracy, ei 0.150000: 95.81 %
#Standard accuracy, ei 0.200000: 94.61 %
#Standard accuracy, ei 0.250000: 92.99 %
#Standard accuracy, ei 0.300000: 90.28 %

pgd pgd
Standard accuracy, ei 0.000000: 94.30 %
Standard accuracy, ei 0.050000: 92.32 %
Standard accuracy, ei 0.100000: 89.89 %
Standard accuracy, ei 0.150000: 87.07 %
Standard accuracy, ei 0.200000: 84.51 %
Standard accuracy, ei 0.250000: 82.17 %
Standard accuracy, ei 0.300000: 79.72 %

Standard accuracy, ei 0.000000: 96.88 %
Standard accuracy, ei 0.050000: 95.77 %
Standard accuracy, ei 0.100000: 94.48 %
Standard accuracy, ei 0.150000: 93.03 %
Standard accuracy, ei 0.200000: 91.79 %
Standard accuracy, ei 0.250000: 90.84 %
Standard accuracy, ei 0.300000: 90.03 %

pgdl2
Standard accuracy, ei 0.000000: 99.23 %
Standard accuracy, ei 0.300000: 98.24 %
Standard accuracy, ei 1.000000: 90.76 %
Standard accuracy, ei 3.000000: 4.08 %
Standard accuracy, ei 10.000000: 0.46 %
Standard accuracy, ei 30.000000: 0.46 %
Standard accuracy, ei 100.000000: 0.46 %

pgdl2
Standard accuracy, ei 0.000000: 85.59 %
Standard accuracy, ei 0.300000: 98.31 %
Standard accuracy, ei 1.000000: 93.19 %
Standard accuracy, ei 3.000000: 91.85 %
Standard accuracy, ei 10.000000: 91.85 %
Standard accuracy, ei 30.000000: 91.85 %
Standard accuracy, ei 100.000000: 91.85 %



#3.  test
ce_model="/home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos/mnist_ce98.69"
python3 attack.py --mode test --loss 0 --gpu 0 --output_name mnist_ce \
--load ${ce_model}


el_model="/home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos/mnist_el99.04"
python3 attack.py --mode test --loss 1 --gpu 1 --output_name mnist_el \
--load ${el_model}




# 4. adv test

ce_model="/home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos/mnist_ce98.69"
for test_atk in pgdl2
do
python3 attack.py --mode adv_test --loss 0 --gpu 4 \
 --test_atk ${test_atk} --load ${ce_model}
done

el_model="/home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos/mnist_el99.04"
for test_atk in pgdl2
do
python3 attack.py --mode adv_test --loss 1 --gpu 5  \
 --test_atk ${test_atk} --load ${el_model}
done


ce_model="/home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos/mnist_ce98.69"
for test_atk in cw
do
python3 attack.py --mode adv_test --loss 0 --gpu 6 \
 --test_atk ${test_atk} --load ${ce_model}
done

el_model="/home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos/mnist_el99.04"
for test_atk in cw # pgd fgsm
do
python3 attack.py --mode adv_test --loss 1 --gpu 7  \
 --test_atk ${test_atk} --load ${el_model}
done



pgd
Standard accuracy, ei 0.000000: 98.69 %
Standard accuracy, ei 0.050000: 93.69 %
Standard accuracy, ei 0.100000: 76.52 %
Standard accuracy, ei 0.150000: 39.69 %
Standard accuracy, ei 0.200000: 13.34 %
Standard accuracy, ei 0.250000: 2.92 %
Standard accuracy, ei 0.300000: 0.53 %


Standard accuracy, ei 0.000000: 99.04 %
Standard accuracy, ei 0.050000: 95.07 %
Standard accuracy, ei 0.100000: 81.32 %
Standard accuracy, ei 0.150000: 54.18 %
Standard accuracy, ei 0.200000: 30.25 %
Standard accuracy, ei 0.250000: 18.12 %
Standard accuracy, ei 0.300000: 11.21 %

fgsm
Standard accuracy, ei 0.000000: 98.69 %
Standard accuracy, ei 0.050000: 94.43 %
Standard accuracy, ei 0.100000: 84.21 %
Standard accuracy, ei 0.150000: 64.80 %
Standard accuracy, ei 0.200000: 44.22 %
Standard accuracy, ei 0.250000: 31.15 %
Standard accuracy, ei 0.300000: 21.72 %


Standard accuracy, ei 0.000000: 99.04 %
Standard accuracy, ei 0.050000: 96.34 %
Standard accuracy, ei 0.100000: 91.09 %
Standard accuracy, ei 0.150000: 79.68 %
Standard accuracy, ei 0.200000: 63.00 %
Standard accuracy, ei 0.250000: 46.50 %
Standard accuracy, ei 0.300000: 34.80 %


ce_model="/home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos/mnist_ce98.69"
for test_atk in aa
do
python3 attack.py --mode adv_test --loss 0 --gpu 4 \
 --test_atk ${test_atk} --load ${ce_model}
done

el_model="/home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos/mnist_el99.04"
for test_atk in aa
do
python3 attack.py --mode adv_test --loss 1 --gpu 5  \
 --test_atk ${test_atk} --load ${el_model}
done



ce_model="/home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos/mnist_ce98.69"
for test_atk in aal2
do
python3 attack.py --mode adv_test --loss 0 --gpu 6 \
 --test_atk ${test_atk} --load ${ce_model}
done

el_model="/home/xxx/encourage_ns/attack/adversarial-attacks-pytorch/demos/mnist_el99.04"
for test_atk in aal2
do
python3 attack.py --mode adv_test --loss 1 --gpu 7  \
 --test_atk ${test_atk} --load ${el_model}
done

