
### Install
```
conda create -n detectron2 python=3.7 -y

conda activate detectron2
conda install pytorch=1.5.0 cudatoolkit=10.1 torchvision -c pytorch

#python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
### Override the files with the files in this directory.

### Training and evaluation scripts
1. For focal loss
```
mkdir encourage_logs
cur_save=focal_loss_alpha050_gamma1
tools/train_net.py \
  --config-file ./configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml \
  --num-gpus 4 --loss_clsz 'el' --base_loss 'fl' --add_loss 'zero'\
  SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 \
  MODEL.RETINANET.FOCAL_LOSS_ALPHA 0.5  MODEL.RETINANET.FOCAL_LOSS_GAMMA 1.0 OUTPUT_DIR encourage_logs/${cur_save}
```
2. For halted focal loss
```
mkdir encourage_logs
beta=0.0
cur_save=ml_beta${beta}_alpha05_gamma1
tools/train_net.py \
  --config-file ./configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml \
  --num-gpus 4 --loss_clsz 'ml' --beta ${beta} \
  SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005 \
  MODEL.RETINANET.FOCAL_LOSS_ALPHA 0.5 MODEL.RETINANET.FOCAL_LOSS_GAMMA 1.0  OUTPUT_DIR encourage_logs/${cur_save}
done
```