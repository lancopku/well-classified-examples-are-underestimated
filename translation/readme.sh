# Install
pip install -e .

# Get data following the preprocess of  examples/translation
# Experiments
## De-En
1. baseline
export CUDA_VISIBLE_DEVICES=0
bonus_gamma=0
for log_end in 0.5
do
for bonus_rho in 1
do
for seed in 1 2 3 4 5
do
save=deen_transformer_fp_unshareemb_bg${bonus_gamma}_le${log_end}_br${bonus_rho}_ellb_elmp_s${seed}
data=/home/xxx/data/data-bin/iwslt14.tokenized.de-en
python3 train.py \
    ${data} \
    --arch transformer_iwslt_de_en \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0001 \
    --criterion encourage_loss --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses \
    --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --update-freq 4 --max-update 20000 --dropout 0.4 --lr 1e-3 \
    --save-dir checkpoint/${save}  --fp16 --seed ${seed}  --el_lb 0.1 --el_mask_pad 1 \
    --log_end ${log_end} --bonus_gamma ${bonus_gamma} --bonus_rho ${bonus_rho} --keep-interval-updates 10 \
    --log-format json --tensorboard-logdir checkpoint/${save}  2>&1 | tee checkpoint/${save}.txt
python3 average_checkpoints.py --inputs checkpoint/${save}  --num-epoch-checkpoints 10 --output checkpoint/${save}/avg_final.pt
python3 generate.py ${data} --path checkpoint/${save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe > results/${save}_avg_final.txt
python3 generate.py ${data} --path checkpoint/${save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${save}_checkpoint_best.txt
done
done
done

2. EL
export CUDA_VISIBLE_DEVICES=0
bonus_gamma=-1
for log_end in 0.5
do
for bonus_rho in 1
do
for seed in 1 2 3 4 5
do
save=deen_transformer_fp_unshareemb_bg${bonus_gamma}_le${log_end}_br${bonus_rho}_ellb_elmp_s${seed}
data=/home/xxx/data/data-bin/iwslt14.tokenized.de-en
python3 train.py \
    ${data} \
    --arch transformer_iwslt_de_en \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0001 \
    --criterion encourage_loss --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses \
    --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --update-freq 4 --max-update 20000 --dropout 0.4 --lr 1e-3 \
    --save-dir checkpoint/${save}  --fp16 --seed ${seed}  --el_lb 0.1 --el_mask_pad 1 \
    --log_end ${log_end} --bonus_gamma ${bonus_gamma} --bonus_rho ${bonus_rho} --keep-interval-updates 10 \
    --log-format json --tensorboard-logdir checkpoint/${save}  2>&1 | tee checkpoint/${save}.txt
python3 average_checkpoints.py --inputs checkpoint/${save}  --num-epoch-checkpoints 10 --output checkpoint/${save}/avg_final.pt
python3 generate.py ${data} --path checkpoint/${save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe > results/${save}_avg_final.txt
python3 generate.py ${data} --path checkpoint/${save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe > results/${save}_checkpoint_best.txt
done
done
done

## Fr-En
1.baseline
export CUDA_VISIBLE_DEVICES=1
bonus_gamma=0
bonus_rho=1
for log_end in 0.5
do
for seed in 1 2 3 4 5
do
save=fren_transformer_shareallemb_fp_bg${bonus_gamma}_le${log_end}_br${bonus_rho}_ellb_elmp_s${seed}
data=/home/xxx/data/data-bin/iwslt17.tokenized.cased.fr-en.spm_bpe16k
python3 train.py \
    ${data} \
    --arch transformer_iwslt_de_en --encoder-layers 2 --decoder-layers 2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.001 \
    --criterion encourage_loss --label-smoothing 0.1 --share-all-embeddings  \
    --max-tokens 4096 \
    --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses \
    --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-update 100000 --dropout 0.3 --lr 0.0005 \
    --save-dir checkpoint/${save}  --fp16 --seed ${seed} \
    --el_lb 0.1 --el_mask_pad 1 \
    --log_end ${log_end} --bonus_gamma ${bonus_gamma} --bonus_rho ${bonus_rho} --keep-interval-updates 10 \
    --log-format json --tensorboard-logdir checkpoint/${save}  --log-interval 1000 2>&1 | tee checkpoint/${save}.txt
python3 average_checkpoints.py --inputs checkpoint/${save}  --num-epoch-checkpoints 10 --output checkpoint/${save}/avg_final.pt
python3 generate.py ${data} --path checkpoint/${save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe sentencepiece > results/${save}_avg_final.txt
python3 generate.py ${data} --path checkpoint/${save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe sentencepiece > results/${save}_checkpoint_best.txt
done
done

2.
export CUDA_VISIBLE_DEVICES=1
bonus_gamma=-1
bonus_rho=1
for log_end in 0.5
do
for seed in 1 2 3 4 5
do
save=fren_transformer_shareallemb_fp_bg${bonus_gamma}_le${log_end}_br${bonus_rho}_ellb_elmp_s${seed}
data=/home/xxx/data/data-bin/iwslt17.tokenized.cased.fr-en.spm_bpe16k
python3 train.py \
    ${data} \
    --arch transformer_iwslt_de_en --encoder-layers 2 --decoder-layers 2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.001 \
    --criterion encourage_loss --label-smoothing 0.1 --share-all-embeddings  \
    --max-tokens 4096 \
    --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses \
    --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-update 100000 --dropout 0.3 --lr 0.0005 \
    --save-dir checkpoint/${save}  --fp16 --seed ${seed} \
    --el_lb 0.1 --el_mask_pad 1 \
    --log_end ${log_end} --bonus_gamma ${bonus_gamma} --bonus_rho ${bonus_rho} --keep-interval-updates 10 \
    --log-format json --tensorboard-logdir checkpoint/${save}  --log-interval 1000 2>&1 | tee checkpoint/${save}.txt
python3 average_checkpoints.py --inputs checkpoint/${save}  --num-epoch-checkpoints 10 --output checkpoint/${save}/avg_final.pt
python3 generate.py ${data} --path checkpoint/${save}/avg_final.pt --batch-size 128 --beam 5 --remove-bpe sentencepiece > results/${save}_avg_final.txt
python3 generate.py ${data} --path checkpoint/${save}/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe sentencepiece > results/${save}_checkpoint_best.txt
done
done
