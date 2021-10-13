## Installation

We provide pip wheels for all major OS/PyTorch/CUDA combinations, see [here](https://pytorch-geometric.com/whl).

To install the binaries for PyTorch 1.7.0, simply run

```sh
$ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
$ pip install torch-geometric
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102`, or `cu110` depending on your PyTorch installation.

|             | `cpu` | `cu92` | `cu101` | `cu102` | `cu110` |
|-------------|-------|--------|---------|---------|---------|
| **Linux**   | ✅    | ✅     | ✅      | ✅      | ✅      |
| **Windows** | ✅    | ❌     | ✅      | ✅      | ✅      |
| **macOS**   | ✅    |        |         |         |         |

## Run
cd the directory of el_graph
```
cd el_graph
```

### Proteins
1. training with CE loss 
```
for time in  1 2 3 4 5
do
python3 graph_classification_val_acc.py \
--runs 50 --gpu 1 --print_every_run 0 \
--conv_method GCNConv --pooling_method SAGPooling --pooling_layer_type GCNConv \
--weight_decay 0.0001 --batch_size 128 --pooling_ratio 0.5  --dataset PROTEINS \
 --bonus_gamma 0
done
```

2. training with Encouraging loss
```
for time in  1 2 3 4 5
do
python3 graph_classification_val_acc.py \
--runs 50 --gpu 1 --print_every_run 0 \
--conv_method GCNConv --pooling_method SAGPooling --pooling_layer_type GCNConv \
--weight_decay 0.0001 --batch_size 128 --pooling_ratio 0.5  --dataset PROTEINS \
 --bonus_gamma -1 --log_end 0.5
done
```
### NCI1
1. training with CE loss
```
for time in  1 2 3 4 5
do
python3 graph_classification_val_acc.py \
--runs 50 --gpu 1 --print_every_run 0 \
--conv_method GCNConv --pooling_method SAGPooling --pooling_layer_type GCNConv \
--weight_decay 0.0001 --batch_size 128 --pooling_ratio 0.5  --dataset NCI1 \
 --bonus_gamma 0
done
```   
2. training with Encouraging loss
```
for time in  1 2 3 4 5
do
python3 graph_classification_val_acc.py \
--runs 50 --gpu 1 --print_every_run 0 \
--conv_method GCNConv --pooling_method SAGPooling --pooling_layer_type GCNConv \
--weight_decay 0.0001 --batch_size 128 --pooling_ratio 0.5  --dataset NCI1 \
 --bonus_gamma -1 --log_end 0.5
done
```

