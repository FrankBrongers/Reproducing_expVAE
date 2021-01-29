#! /bin/sh
printf "Training FactorVAEs\n"

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
        --print_iter 10000 --gamma 40 --name gamma40 \
        --dis_score --z_dim 32

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
        --print_iter 10000 --gamma 40 --name gamma40 \
        --dis_score --z_dim 32 --seed 1

python main.py --dataset dsprites --ckpt_save_iter 300000 --max_iter 300000 \
        --print_iter 10000 --gamma 40 --name gamma40 \
        --dis_score --z_dim 32 --seed 2


printf "\nTraining AD-FactorVAEs\n"
printf "\nlambda=1\n"

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda1_gamma40 \
       --ad_loss --dis_score --z_dim 32

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda1_gamma40 \
       --ad_loss --dis_score --z_dim 32 --seed 1

python main.py --dataset dsprites --ckpt_save_iter 300000 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda1_gamma40 \
       --ad_loss --dis_score --z_dim 32 --seed 2


printf "\nlambda=1, convlayer=3\n"

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
      --print_iter 10000 --gamma 40 --name conv3_lambda1_gamma40 \
      --ad_loss --dis_score --z_dim 32 --target_layer 4

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
      --print_iter 10000 --gamma 40 --name conv3_lambda1_gamma40 \
      --ad_loss --dis_score --z_dim 32 --seed 1 --target_layer 4

python main.py --dataset dsprites --ckpt_save_iter 300000 --max_iter 300000 \
      --print_iter 10000 --gamma 40 --name conv3_lambda1_gamma40 \
      --ad_loss --dis_score --z_dim 32 --seed 2 --target_layer 4


printf "\nlambda=20\n"

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda20_gamma40 \
       --ad_loss --dis_score --z_dim 32 --lamb 20

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda20_gamma40 \
       --ad_loss --dis_score --z_dim 32 --seed 1 --lamb 20

python main.py --dataset dsprites --ckpt_save_iter 300000 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda20_gamma40 \
       --ad_loss --dis_score --z_dim 32 --seed 2 --lamb 20


printf "\nlambda=40\n"

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda40_gamma40 \
       --ad_loss --dis_score --z_dim 32 --lamb 40

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda40_gamma40 \
       --ad_loss --dis_score --z_dim 32 --seed 1 --lamb 40

python main.py --dataset dsprites --ckpt_save_iter 300000 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda40_gamma40 \
       --ad_loss --dis_score --z_dim 32 --seed 2 --lamb 40


printf "\nlambda=80\n"

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda80_gamma40 \
       --ad_loss --dis_score --z_dim 32 --lamb 80

python main.py --dataset dsprites --ckpt_save_iter 300002 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda80_gamma40 \
       --ad_loss --dis_score --z_dim 32 --seed 1 --lamb 80

python main.py --dataset dsprites --ckpt_save_iter 300000 --max_iter 300000 \
       --print_iter 10000 --gamma 40 --name lambda80_gamma40 \
       --ad_loss --dis_score --z_dim 32 --seed 2 --lamb 80
