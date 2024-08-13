#!/bin/bash
#SBATCH -p gpu1
#SBATCH --gpus=1
#SBATCH -w aiwkr3


#module load cuda/11.7.0
#module load singularity/3.11.0
module load cuda/11.8.0
module load anaconda/anaconda3-2022.10

source activate py310t2cu118

# 新      v202406052040      画图                            2024.6.16 18：10

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log
fi

if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/draw" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/draw
fi
if [ ! -d "/mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/draw/log_06161810_55" ]; then
    mkdir /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/draw/log_06161810_55
fi

 python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_draw.py \
  > /mnt/nfs/data/home/1120231455/home/fuy/python/time_series/run_log/draw/log_06161810_55/'0'_$model_name'_'draw'_'0.01.log 2>&1


















if false;then

seq_len=96



use_x_mark_enc=0
front_use_decomp=0



use_fourier=1
use_space_merge=0
use_conv=0


if [ ! -d "./log_06072122_55" ]; then
    mkdir ./log_06072122_55
fi



if [ ! -d "./log_06072122_55/electricity" ]; then
    mkdir ./log_06072122_55/electricity
fi

if [ ! -d "./log_06072122_55/ETTm1" ]; then
    mkdir ./log_06072122_55/ETTm1
fi
if [ ! -d "./log_06072122_55/ETTh1" ]; then
    mkdir ./log_06072122_55/ETTh1
fi
if [ ! -d "./log_06072122_55/ETTm2" ]; then
    mkdir ./log_06072122_55/ETTm2
fi

if [ ! -d "./log_06072122_55/ETTh2" ]; then
    mkdir ./log_06072122_55/ETTh2
fi


if [ ! -d "./log_06072122_55/Exchange" ]; then
    mkdir ./log_06072122_55/Exchange
fi

if [ ! -d "./log_06072122_55/Solar" ]; then
    mkdir ./log_06072122_55/Solar
fi

if [ ! -d "./log_06072122_55/weather" ]; then
    mkdir ./log_06072122_55/weather
fi

if [ ! -d "./log_06072122_55/Traffic" ]; then
    mkdir ./log_06072122_55/Traffic
fi

if [ ! -d "./log_06072122_55/PEMS03" ]; then
    mkdir ./log_06072122_55/PEMS03
fi

if [ ! -d "./log_06072122_55/PEMS04" ]; then
    mkdir ./log_06072122_55/PEMS04
fi

if [ ! -d "./log_06072122_55/PEMS07" ]; then
    mkdir ./log_06072122_55/PEMS07
fi
if [ ! -d "./log_06072122_55/PEMS08" ]; then
    mkdir ./log_06072122_55/PEMS08
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





ETTm2_cut_freq=10



#  h1 2分解核  dmodel=dff=256

for pred_len in 96  192 336 720; do
 python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
      --model micn \
      --root_path ETT-small \
      --mode regre \
      --data ETTh1 \
      --freq h \
      --features M \
      --e_layers 2 \
      --d_layers 1 \
      --d_model 256 \
      --d_ff 256 \
      --itr 1 \
      --x_enc_len 7 \
      --x_mark_len $ETTh1_x_mark_len \
      --seq_len 96 \
      --pred_len $pred_len \
      --down_sampling_layers 3 \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --channel_independence 1 \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_x_mark_enc $use_x_mark_enc \
      --use_conv $use_conv \
  > log_06072122_55/ETTh1/'0'_$model_name'_'ETTh1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
done





#  h2 2分解核  d_model= d_ff = 128

for pred_len in 96  192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --data ETTh2 \
        --freq h \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len $ETTh2_x_mark_len \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
        --use_conv $use_conv \
    > log_06072122_55/ETTh2/'0'_$model_name'_'ETTh2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done






# m1 d_model= d_ff = 128

  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --model micn \
        --root_path ETT-small \
        --mode regre \
        --freq t \
        --data ETTm1 \
        --features M \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --x_enc_len 7 \
        --x_mark_len $ETTm1_x_mark_len \
        --seq_len 96 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06072122_55/ETTm1/'0'_$model_name'_'ETTm1'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done



#  m2 2分解核

# m2 d_model= d_ff = 128
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --root_path ETT-small \
        --model micn \
        --mode regre \
        --data ETTm2 \
        --features M \
        --d_layers 1 \
        --freq t \
        --seq_len 96 \
        --pred_len $pred_len \
        --d_model 128 \
        --d_ff 128 \
        --x_enc_len 7 \
        --x_mark_len $ETTm2_x_mark_len \
        --e_layers 2 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06072122_55/ETTm2/'0'_$model_name'_'ETTm2'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done













# electricity d_model=d_ff=512

 for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --model micn \
        --root_path electricity \
        --mode regre \
        --data ECL \
        --freq h \
        --features M \
        --e_layers 3 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 512 \
        --batch_size 16 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --learning_rate 0.0005 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06072122_55/electricity/'0'_$model_name'_'electricity'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done








# Exchange d_model=d_ff=128
  for pred_len in 96 192 336 720; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --model micn \
        --root_path exchange_rate \
        --mode regre \
        --data Exchange \
        --features M \
        --e_layers 2 \
        --freq d \
        --d_layers 1 \
        --d_model 128 \
        --d_ff 128 \
        --seq_len 96 \
        --itr 1 \
        --pred_len $pred_len \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $Exchange_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06072122_55/Exchange/'0'_$model_name'_'Exchange'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done











#  weather 3分解核



# weather d_ff=d_ff=512
  for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --model micn \
        --root_path weather \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --d_layers 1 \
        --e_layers 3 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $weather_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06072122_55/weather/'0'_$model_name'_'weather'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done






#PEMS03
for pred_len in 12 24 48 96 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --model micn \
        --root_path PEMS \
        --data PEMS03 \
        --freq t \
        --features M \
        --d_layers 1 \
        --e_layers 4 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --learning_rate 0.001 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS03_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06072122_55/PEMS03/'0'_$model_name'_'PEM03'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done


#PEMS04
for pred_len in 12 24 48 96 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --model micn \
        --root_path PEMS \
        --data PEMS04 \
        --freq t \
        --features M \
        --d_layers 1 \
        --e_layers 4 \
        --d_ff 1024 \
        --d_model 1024 \
        --seq_len 96 \
        --pred_len $pred_len \
        --learning_rate 0.0005 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS07_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06072122_55/PEMS04/'0'_$model_name'_'PEM04'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
#PEMS07
for pred_len in 12 24 48 96 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --model micn \
        --root_path PEMS \
        --data PEMS07 \
        --freq t \
        --features M \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --learning_rate 0.001 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS07_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06072122_55/PEMS07/'0'_$model_name'_'PEM07'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
#PEMS08
for pred_len in 12 24 48 96 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --model micn \
        --root_path PEMS \
        --data PEMS08 \
        --freq t \
        --features M \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS08_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06072122_55/PEMS08/'0'_$model_name'_'PEM08'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done

# Solar
for pred_len in 96 192 336 720 ; do
   python -u  /mnt/nfs/data/home/1120231455/home/fuy/python/test_MICN_1455/run.py \
        --model micn \
        --root_path Solar \
        --mode regre \
        --data Solar \
        --freq t \
        --features M \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 512 \
        --d_model 512 \
        --seq_len 96 \
        --pred_len $pred_len \
        --learning_rate 0.0005 \
        --itr 1 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $Solar_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_x_mark_enc $use_x_mark_enc \
    > log_06072122_55/Solar/'0'_$model_name'_'Solar'_'$seq_len'_'$pred_len'_'0.01.log 2>&1
  done
fi