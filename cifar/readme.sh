# experiments  LR and bsz can be scaled according to the linear scaling rule to maximize the utilization of gpu
1. Resnt50
#1.0 baseline
gpu=0
model='resnet50'
bg=0 # set to baseline
le=1
br=1
for data in cifar10 cifar100
do
for t in 1 2 3 4 5
do
cur_save=${data}_${model}_bg${bg}_le${le}_br${br}_t${t}
python3 train.py -c configs/${data}.yaml --gpu ${gpu}  --log ${cur_save} --model ${model} \
--batch_size 64 --lr 0.025 --bonus_gamma ${bg} --log_end ${le} --bonus_rho ${br}
done
done
#1.1 EL
gpu=1
model='resnet50'
bg=-1
br=1
le=0.75
for data in cifar10 cifar100
do
for t in 1 2 3 4 5
do
cur_save=${data}_${model}_bg${bg}_le${le}_br${br}_t${t}
python3 train.py -c configs/${data}.yaml --gpu ${gpu}  --log ${cur_save} --model ${model} \
--batch_size 64 --lr 0.025 --bonus_gamma ${bg} --log_end ${le} --bonus_rho ${br}
done
done

2. EffficientNet
2.0 baseline
gpu=2
model='efficientnet_b0'
bg=0
br=1
le=1
for data in cifar10 cifar100
do
for t in 1 2 3 4 5
do
cur_save=${data}_${model}_bg${bg}_le${le}_br${br}_t${t}
python3 train.py -c configs/${data}.yaml --gpu ${gpu}  --log ${cur_save} --model ${model} \
--bonus_gamma ${bg} --log_end ${le} --bonus_rho ${br}
done
done


2.1 EL
gpu=3
model='efficientnet_b0'
bg=-1
br=1
le=0.75
for data in cifar10 cifar100
do
for t in 1 2 3 4 5
do
cur_save=${data}_${model}_bg${bg}_le${le}_br${br}_t${t}
python3 train.py -c configs/${data}.yaml --gpu ${gpu}  --log ${cur_save} --model ${model} \
--bonus_gamma ${bg} --log_end ${le} --bonus_rho ${br}
done
done


