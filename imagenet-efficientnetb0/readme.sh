# Install
clone the code from pytorch-image-models, install required packages, and overwrite files

# Experiments
1. baseline
export CUDA_VISIBLE_DEVICES=0,1
bonus_gamma=0
log_end=1
bonus_rho=1
save_dir=eb0_bg${bonus_gamma}_le${log_end}_br${bonus_rho}
python3 -m torch.distributed.launch --nproc_per_node=2 train_el.py /data/datasets/Img  --model efficientnet_b0 -b 384 --sched step --epochs 450 --decay-epochs 2.4 \
--decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 \
--model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048 \
--dist_url 'tcp://127.0.0.1:23472' --bonus_gamma ${bonus_gamma} --log_end ${log_end} --bonus_rho ${bonus_rho}  --output ${save_dir}

2. EL
export CUDA_VISIBLE_DEVICES=0,1
bonus_gamma=-1
log_end=0.75
bonus_rho=1
save_dir=eb0_bg${bonus_gamma}_le${log_end}_br${bonus_rho}
python3 -m torch.distributed.launch --nproc_per_node=2 train_el.py /data/datasets/Img  --model efficientnet_b0 -b 384 --sched step --epochs 450 --decay-epochs 2.4 \
--decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 \
--model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048 \
--dist_url 'tcp://127.0.0.1:23472' --bonus_gamma ${bonus_gamma} --log_end ${log_end} --bonus_rho ${bonus_rho}  --output ${save_dir}
