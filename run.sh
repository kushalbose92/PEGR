
# TUDatasets --- Molecular graphs

# ---------------\

# for GCN

# python -u tu_datasets_main.py --dataset 'ENZYMES' --train_lr 0.001 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 8 --rewiring None --model gcn --alpha 0.0 --device 'cuda:0' | tee output/enzymes.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 8 --rewiring fosr --model gcn --alpha 2.0 --device 'cuda:0' | tee output/mutag.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 4 --mlp_layers 1 --hidden_dim 64 --train_iter 50 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0001 --th 0.0 --num_splits 5 --batch_size 64 --rewiring gtr --model gin --alpha 2.0 --device 'cuda:0' | tee output/proteins.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 8 --rewiring sdrf --model gcn --alpha 1.0 --device 'cuda:0' | tee output/collab.txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 1 --rewiring sdrf --model gcn --alpha 2.0 --device 'cuda:0' | tee output/reddit-binary.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 16 --rewiring fosr --model gcn --alpha 2.0 --device 'cuda:0' | tee output/imdb-binary.txt


# --------------------------

#  For GIN

# python -u tu_datasets_main.py --dataset 'ENZYMES' --train_lr 0.001 --seed 0 --num_layers 3 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 8 --rewiring gtr --model gin --alpha 0.0 --device 'cuda:0' | tee output/enzymes.txt

# python -u tu_datasets_main.py --dataset 'MUTAG' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 8 --rewiring gtr --model gin --alpha 0.0 --device 'cuda:0' | tee output/mutag.txt

# python -u tu_datasets_main.py --dataset 'PROTEINS' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th 0.0 --num_splits 25 --batch_size 8 --rewiring gtr --model gin --alpha 0.0 --device 'cuda:0' | tee output/proteins.txt

# python -u tu_datasets_main.py --dataset 'COLLAB' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 8 --rewiring gtr --model gin --alpha 0.0 --device 'cuda:0' | tee output/collab.txt

# python -u tu_datasets_main.py --dataset 'REDDIT-BINARY' --train_lr 0.001 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 1 --rewiring gtr --model gin --alpha 0.0 --device 'cuda:0' | tee output/reddit-binary.txt

# python -u tu_datasets_main.py --dataset 'IMDB-BINARY' --train_lr 0.01 --seed 0 --num_layers 2 --mlp_layers 1 --hidden_dim 64 --train_iter 20 --test_iter 1 --use_saved_model False --dropout 0.0 --train_w_decay 0.0005 --th None --num_splits 25 --batch_size 16 --rewiring None --model gin --alpha 0.0 --device 'cuda:0' | tee output/imdb-binary.txt


