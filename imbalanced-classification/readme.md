## Setup
* Clone the code from the [classifier-balancing](https://github.com/facebookresearch/classifier-balancing)
* Download data and install environments following their readme
* overwrite files with files in this directory
## Experiments
### CE loss
1. baseline
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
data=iNaturalist18
uniform_courage=1
beta=0.9999
save=${data}_el
for time in 1
do
bonus_start=0
for bg in -1
do
cur_save=${save}_e200_CE_t${time}
python3 main.py --cfg config/${data}/feat_uniform.yaml  --num_epochs 200  -ds 10000 --beta ${beta}  \
--loss_type 'courage'  -cw 'same' --courage_by_weight 1 --uniform_courage ${uniform_courage} --bonus_start ${bonus_start} -bg ${bg} \
--log_dir ./logs/${data}/models/${cur_save} \
2>&1 | tee -a ${cur_save}.txt
done
done
```
2. add normal bonus
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
data=iNaturalist18
uniform_courage=1
beta=0.9999
save=${data}_el
for time in 1
do
bonus_start=0
for bg in -1
do
cur_save=${save}_e200_el_bg${bg}_bs${bonus_start}_beta${beta}_t${time}
python3 main.py --cfg config/${data}/feat_uniform.yaml  --num_epochs 200  -ds 0 --beta ${beta}  \
--loss_type 'courage'  -cw 'same' --courage_by_weight 1 --uniform_courage ${uniform_courage} --bonus_start ${bonus_start} -bg ${bg} \
--log_dir ./logs/${data}/models/${cur_save} \
2>&1 | tee -a ${cur_save}.txt
done
done
```
3. add aggressive bonus
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
data=iNaturalist18
uniform_courage=1
beta=0.9999
save=${data}_el
for time in 1
do
bonus_start=0.5
for bg in -1
do
cur_save=${save}_e200_el_bg${bg}_bs${bonus_start}_beta${beta}_t${time}
python3 main.py --cfg config/${data}/feat_uniform.yaml  --num_epochs 200  -ds 0 --beta ${beta}  \
--loss_type 'courage'  -cw 'same' --courage_by_weight 1 --uniform_courage ${uniform_courage} --bonus_start ${bonus_start} -bg ${bg} \
--log_dir ./logs/${data}/models/${cur_save} \
2>&1 | tee -a ${cur_save}.txt
done
done
```
### Defer the re-weighting

1. baseline
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 
data=iNaturalist18
save=${data}
epochs=200
start=180
for time in 1 2 3
do
cur_save=${save}_e${epochs}_del_drw${start}_t${time}
python3 main.py --cfg config/${data}/feat_uniform.yaml  --num_epochs ${epochs}  -ds ${start} -es  10000  \
--loss_type 'courage'  -cw 'same' --courage_by_weight 1 --base_loss 'DRW' \
--log_dir ./logs/${data}/models/${cur_save} \
2>&1 | tee -a ${cur_save}.txt
done
```      
2. add normal bonus
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 
data=iNaturalist18
save=${data}
epochs=200
start=180
for time in 1 2 3
do
cur_save=${save}_e${epochs}_del_drw${start}_t${time}
python3 main.py --cfg config/${data}/feat_uniform.yaml  --num_epochs ${epochs}  -ds ${start} \
--loss_type 'courage'  -cw 'same' --courage_by_weight 1 --base_loss 'DRW' \
--log_dir ./logs/${data}/models/${cur_save} \
2>&1 | tee -a ${cur_save}.txt
done
```   

### Decoupling
1. on baseline
```
#iNaturalist18_CE is the directory to the trained model with the CE loss
python3 main.py --cfg ./config/iNaturalist18/cls_crt.yaml --model_dir './logs/iNaturalist18/models/iNaturalist18_CE' \
--log_dir ./logs/iNaturalist18/clslearn/crt_CE
```
2. on our method, which learns representations with EL in the first stage
```
#iNaturalist18_CE is the directory to the trained model with the EL loss
python3 main.py --cfg ./config/iNaturalist18/cls_crt.yaml --model_dir './logs/iNaturalist18/models/iNaturalist18_EL' \
--log_dir ./logs/iNaturalist18/clslearn/crt_EL
```
