train the ResNet50 on ImageNet


# baseline
export CUDA_VISIBLE_DEVICES=0,1
bonus_gamma=0
log_end=1
for seed in 1 2 3 4 5
do
save_dir=resnet50_313_bg${bonus_gamma}_le${log_end}_s${seed}
python3 main.py -a resnet50 --dist-url 'tcp://127.0.0.1:23472' --dist-backend 'nccl' --multiprocessing-distributed -p 1000 \
--world-size 1 --seed ${seed} --rank 0 /data/datasets/Img --bonus_gamma ${bonus_gamma} --log_end ${log_end} --save_dir ${save_dir} 2>&1 | tee -a  ${save_dir}_print.txt
done

# EL with conservative bonus
export CUDA_VISIBLE_DEVICES=2,3
bonus_gamma=-1
log_end=0.75
for seed in 1 2 3 4 5
do
save_dir=resnet50_313_bg${bonus_gamma}_le${log_end}_s${seed}
python3 main.py -a resnet50 --dist-url 'tcp://127.0.0.1:23472' --dist-backend 'nccl' --multiprocessing-distributed -p 1000 \
--world-size 1 --seed ${seed} --rank 0 /data/datasets/Img --bonus_gamma ${bonus_gamma} --log_end ${log_end} --save_dir ${save_dir} 2>&1 | tee -a  ${save_dir}_print.txt
done


