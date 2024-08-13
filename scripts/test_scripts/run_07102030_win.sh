source activate py310t2cu118

# 绘制不同的图

use_x_mark_enc=0
front_use_decomp=0
use_space_merge=0


use_fourier=1
use_conv=1

use_origin_seq=1
use_linear=1
use_relu=1


seq_len=96
if [ ! -d "../../run_log" ]; then
    mkdir ../../run_log
fi
if [ ! -d "../../run_log/log_07102030_win" ]; then
    mkdir ../../run_log/log_07102030_win
fi
if [ ! -d "../../run_log/log_07102030_win/ETTm1" ]; then
    mkdir ../../run_log/log_07102030_win/ETTm1
fi
if [ ! -d "../../run_log/log_07102030_win/ETTh1" ]; then
    mkdir ../../run_log/log_07102030_win/ETTh1
fi
if [ ! -d "../../run_log/log_07102030_win/ETTm2" ]; then
    mkdir ../../run_log/log_07102030_win/ETTm2
fi

if [ ! -d "../../run_log/log_07102030_win/ETTh2" ]; then
    mkdir ../../run_log/log_07102030_win/ETTh2
fi
if [ ! -d "../../run_log/log_07102030_win/electricity" ]; then
    mkdir ../../run_log/log_07102030_win/electricity
fi

if [ ! -d "../../run_log/log_07102030_win/Exchange" ]; then
    mkdir ../../run_log/log_07102030_win/Exchange
fi

if [ ! -d "../../run_log/log_07102030_win/Solar" ]; then
    mkdir ../../run_log/log_07102030_win/Solar
fi

if [ ! -d "../../run_log/log_07102030_win/weather" ]; then
    mkdir ../../run_log/log_07102030_win/weather
fi

if [ ! -d "../../run_log/log_07102030_win/Traffic" ]; then
    mkdir ../../run_log/log_07102030_win/Traffic
fi

if [ ! -d "../../run_log/log_07102030_win/PEMS03" ]; then
    mkdir ../../run_log/log_07102030_win/PEMS03
fi

if [ ! -d "../../run_log/log_07102030_win/PEMS04" ]; then
    mkdir ../../run_log/log_07102030_win/PEMS04
fi

if [ ! -d "../../run_log/log_07102030_win/PEMS07" ]; then
    mkdir ../../run_log/log_07102030_win/PEMS07
fi
if [ ! -d "../../run_log/log_07102030_win/PEMS08" ]; then
    mkdir ../../run_log/log_07102030_win/PEMS08
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




# 原始序列1 + 全连接1 + relu1
use_origin_seq=1
use_linear=1
use_relu=1

#weather
for down_sampling_layer in 3 4 5 6 7 8 9 10; do

  for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_WTH_'96'_'$pred_len \
        --model TaperFourierFormer \
        --root_path weather \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 512 \
        --d_model 512 \
        --batch_size 16 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $weather_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > ../../run_log/log_07102030_win/weather/'0_TaperFourierFormer_'weather'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
done
# 原始序列1 + 全连接1 + relu1
use_origin_seq=1
use_linear=1
use_relu=1

#weather
for down_sampling_layer in 0 1 2 11 15 20 128; do

  for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_WTH_'96'_'$pred_len \
        --model TaperFourierFormer \
        --root_path weather \
        --mode regre \
        --data WTH \
        --features M \
        --freq t \
        --d_layers 1 \
        --e_layers 2 \
        --d_ff 512 \
        --d_model 512 \
        --batch_size 16 \
        --seq_len 96 \
        --pred_len $pred_len \
        --itr 1 \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $weather_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_origin_seq $use_origin_seq \
        --use_linear $use_linear \
        --use_relu $use_relu \
      > ../../run_log/log_07102030_win/weather/'0_TaperFourierFormer_'weather'_'$seq_len'_'$pred_len'_down_sampling_layer='$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
done