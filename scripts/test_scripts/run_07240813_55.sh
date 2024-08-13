#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH -w aiwkr2


#module load cuda/11.7.0
#module load singularity/3.11.0
module load cuda/11.8.0
module load anaconda/anaconda3-2022.10

source activate py310t2cu118

use_x_mark_enc=0
front_use_decomp=0
use_space_merge=0


use_fourier=1
use_conv=1

use_origin_seq=1
use_linear=1
use_relu=1


seq_len=96
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTm1" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTm1
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTh1" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTh1
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTm2" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTm2
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTh2" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTh2
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/electricity" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/electricity
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/Exchange" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/Exchange
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/Solar" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/Solar
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/weather" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/weather
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/Traffic" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/Traffic
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/PEMS03" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/PEMS03
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/PEMS04" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/PEMS04
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/PEMS07" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/PEMS07
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/PEMS08" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/PEMS08
fi
#singularity exec --nv /mnt/nfs/data/home/1120231455/home/fuy/fuypycharm1_1.sif  nvidia-smi;\

date_ETTh1=ETTh1
date_ETTh2=ETTh2
date_ETTm1=ETTm1
date_ETTm2=ETTm2
date_exchange_rate=exchange_rate
date_national_illness=national_illness
date_weather=weather
# 分解核除了ill是 19 13，其他都是13 17  且必须为奇数
decomp_kernel=(17 49 81)
electricity_x_mark_len=4
weather_x_mark_len=5
Exchange_x_mark_len=3
illness_x_mark_len=3
traffic_x_mark_len=4
ETTh1_x_mark_len=4
ETTh2_x_mark_len=4
ETTm2_x_mark_len=5
ETTm1_x_mark_len=5

Solar_x_mark_len=1
PEMS03_x_mark_len=1
PEMS04_x_mark_len=1
PEMS07_x_mark_len=1
PEMS08_x_mark_len=1
if false;then
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=5
train_epochs=30
patience=10
learning_rate=0.001
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --model_id NewModelTimesNet'_ETTh1_'96'_'$pred_len \
        --root_path ETT-small \
        --mode regre \
        --data ETTh1 \
        --freq h \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --train_epochs $train_epochs \
        --patience $patience \
        --learning_rate $learning_rate \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTh1/'test0_NewModelTimesNet_'ETTh1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done
fi

use_atten=1
use_fourier_att=0
#  h1 2分解核  dmodel=dff=256
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=5
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --model_id NewModelTimesNet'_ETTh1_'96'_'$pred_len \
        --root_path ETT-small \
        --mode regre \
        --data ETTh1 \
        --freq h \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTh1/'0_NewModelTimesNet_'ETTh1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done


#  h2 ok
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=4
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path ETT-small \
        --model_id NewModelTimesNet'_ETTh2_'96'_'$pred_len \
        --mode regre \
        --data ETTh2 \
        --freq h \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTh2/'0_NewModelTimesNet_'ETTh2'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done




# m1 d_model= d_ff = 128
#  m1 ok
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=3
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path ETT-small \
        --model_id NewModelTimesNet'_ETTm1_'96'_'$pred_len \
        --mode regre \
        --data ETTm1 \
        --freq t \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k  \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTm1/'0_NewModelTimesNet_'ETTm1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done


#  m2 2分解核
# m2   ok

layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=2
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path ETT-small \
        --model_id NewModelTimesNet'_ETTm2_'96'_'$pred_len \
        --mode regre \
        --data ETTm2 \
        --freq t \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTm2/'0_NewModelTimesNet_'ETTm2'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done

if false;then
#ECL
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=2
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path electricity \
        --model_id NewModelTimesNet'_ECL_'96'_'$pred_len \
        --mode regre \
        --data ECL \
        --freq t \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/electricity/'0_NewModelTimesNet_'electricity'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done

fi


# Exchange d_model=d_ff=128  ok

layer_nums=2
d_model=128
d_ff=128
down_sampling_layers=3
use_origin_seq=1
k=5
learning_rate=0.0005

for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path exchange_rate \
        --model_id NewModelTimesNet'_Exchange_'96'_'$pred_len \
        --mode regre \
        --data Exchange \
        --freq d \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --learning_rate $learning_rate \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/Exchange/'0_NewModelTimesNet_'Exchange'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done


#weather

layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=4
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path weather \
        --model_id NewModelTimesNet'_WTH_'96'_'$pred_len \
        --mode regre \
        --data WTH \
        --freq d \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/weather/'0_NewModelTimesNet_'weather'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done

layer_nums=2
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=4
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path weather \
        --model_id NewModelTimesNet'_WTH_'96'_'$pred_len \
        --mode regre \
        --data WTH \
        --freq d \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/weather/'0_NewModelTimesNet_'weather'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done
































#  -------------




use_atten=0
use_fourier_att=1
#  h1 2分解核  dmodel=dff=256
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=5
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --model_id NewModelTimesNet'_ETTh1_'96'_'$pred_len \
        --root_path ETT-small \
        --mode regre \
        --data ETTh1 \
        --freq h \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTh1/'0_NewModelTimesNet_'ETTh1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done


#  h2 ok
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=4
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path ETT-small \
        --model_id NewModelTimesNet'_ETTh2_'96'_'$pred_len \
        --mode regre \
        --data ETTh2 \
        --freq h \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTh2/'0_NewModelTimesNet_'ETTh2'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done




# m1 d_model= d_ff = 128
#  m1 ok
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=3
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path ETT-small \
        --model_id NewModelTimesNet'_ETTm1_'96'_'$pred_len \
        --mode regre \
        --data ETTm1 \
        --freq t \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k  \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTm1/'0_NewModelTimesNet_'ETTm1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done


#  m2 2分解核
# m2   ok

layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=2
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path ETT-small \
        --model_id NewModelTimesNet'_ETTm2_'96'_'$pred_len \
        --mode regre \
        --data ETTm2 \
        --freq t \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/ETTm2/'0_NewModelTimesNet_'ETTm2'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done

if false;then
#ECL
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=2
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path electricity \
        --model_id NewModelTimesNet'_ECL_'96'_'$pred_len \
        --mode regre \
        --data ECL \
        --freq t \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/electricity/'0_NewModelTimesNet_'electricity'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done

fi


# Exchange d_model=d_ff=128  ok

layer_nums=2
d_model=128
d_ff=128
down_sampling_layers=3
use_origin_seq=1
k=5
learning_rate=0.0005

for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path exchange_rate \
        --model_id NewModelTimesNet'_Exchange_'96'_'$pred_len \
        --mode regre \
        --data Exchange \
        --freq d \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --learning_rate $learning_rate \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/Exchange/'0_NewModelTimesNet_'Exchange'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done


#weather

layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=4
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path weather \
        --model_id NewModelTimesNet'_WTH_'96'_'$pred_len \
        --mode regre \
        --data WTH \
        --freq d \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/weather/'0_NewModelTimesNet_'weather'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done

layer_nums=2
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=4
for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model NewModelTimesNet \
        --root_path weather \
        --model_id NewModelTimesNet'_WTH_'96'_'$pred_len \
        --mode regre \
        --data WTH \
        --freq d \
        --features M \
        --itr 1 \
        --seq_len 96 \
        --pred_len $pred_len \
        --des Exp \
        --n_heads 16 \
        --batch_size 32 \
        --layer_nums $layer_nums \
        --down_sampling_layers $down_sampling_layers \
        --d_model $d_model \
        --d_ff $d_ff \
        --use_origin_seq $use_origin_seq \
        --k $k \
        --use_atten $use_atten \
        --use_fourier_att $use_fourier_att \
        > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/log_07240813_55/weather/'0_NewModelTimesNet_'weather'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_use_atten='$use_atten'_use_fourier_att='$use_fourier_att'_'0.01.log 2>&1
  done