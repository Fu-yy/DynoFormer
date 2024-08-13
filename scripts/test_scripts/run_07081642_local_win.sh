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
if [ ! -d "../../run_log/log_07081643_local_win" ]; then
    mkdir ../../run_log/log_07081643_local_win
fi
if [ ! -d "../../run_log/log_07081643_local_win/ETTm1" ]; then
    mkdir ../../run_log/log_07081643_local_win/ETTm1
fi
if [ ! -d "../../run_log/log_07081643_local_win/ETTh1" ]; then
    mkdir ../../run_log/log_07081643_local_win/ETTh1
fi
if [ ! -d "../../run_log/log_07081643_local_win/ETTm2" ]; then
    mkdir ../../run_log/log_07081643_local_win/ETTm2
fi

if [ ! -d "../../run_log/log_07081643_local_win/ETTh2" ]; then
    mkdir ../../run_log/log_07081643_local_win/ETTh2
fi
if [ ! -d "../../run_log/log_07081643_local_win/electricity" ]; then
    mkdir ../../run_log/log_07081643_local_win/electricity
fi

if [ ! -d "../../run_log/log_07081643_local_win/Exchange" ]; then
    mkdir ../../run_log/log_07081643_local_win/Exchange
fi

if [ ! -d "../../run_log/log_07081643_local_win/Solar" ]; then
    mkdir ../../run_log/log_07081643_local_win/Solar
fi

if [ ! -d "../../run_log/log_07081643_local_win/weather" ]; then
    mkdir ../../run_log/log_07081643_local_win/weather
fi

if [ ! -d "../../run_log/log_07081643_local_win/Traffic" ]; then
    mkdir ../../run_log/log_07081643_local_win/Traffic
fi

if [ ! -d "../../run_log/log_07081643_local_win/PEMS03" ]; then
    mkdir ../../run_log/log_07081643_local_win/PEMS03
fi

if [ ! -d "../../run_log/log_07081643_local_win/PEMS04" ]; then
    mkdir ../../run_log/log_07081643_local_win/PEMS04
fi

if [ ! -d "../../run_log/log_07081643_local_win/PEMS07" ]; then
    mkdir ../../run_log/log_07081643_local_win/PEMS07
fi
if [ ! -d "../../run_log/log_07081643_local_win/PEMS08" ]; then
    mkdir ../../run_log/log_07081643_local_win/PEMS08
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




# 原始序列0 + 全连接1 + relu1
use_origin_seq=0
use_linear=1
use_relu=1
#exchange
down_sampling_layer=3

for pred_len in 96 192 336 720; do
 python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
      --model TaperFourierFormer \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --x_mark_len $Exchange_x_mark_len \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_x_mark_enc $use_x_mark_enc \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
      --use_origin_seq $use_origin_seq \
  > ../../run_log/log_07081643_local_win/Exchange/'0_TaperFourierFormer_'Exchange'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done

# 原始序列0 + 全连接0 + relu0
use_origin_seq=0
use_linear=0
use_relu=0
for pred_len in 96 192 336 720; do
 python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
      --model TaperFourierFormer \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --x_mark_len $Exchange_x_mark_len \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_x_mark_enc $use_x_mark_enc \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
      --use_origin_seq $use_origin_seq \
  > ../../run_log/log_07081643_local_win/Exchange/'0_TaperFourierFormer_'Exchange'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done
# 原始序列0 + 全连接1 + relu0
use_origin_seq=0
use_linear=1
use_relu=0
for pred_len in 96 192 336 720; do
 python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
      --model TaperFourierFormer \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --x_mark_len $Exchange_x_mark_len \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_x_mark_enc $use_x_mark_enc \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
      --use_origin_seq $use_origin_seq \
  > ../../run_log/log_07081643_local_win/Exchange/'0_TaperFourierFormer_'Exchange'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done

if false;then
# 原始序列1 + 全连接0 + relu0
use_origin_seq=1
use_linear=0
use_relu=0
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path ETT-small \
        --model_id TaperFourierFormer'_ETTh2_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --use_origin_seq $use_origin_seq \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTh2/'0_TaperFourierFormer_'ETTh2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
# 原始序列1 + 全连接1 + relu1
use_origin_seq=1
use_linear=1
use_relu=1
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path ETT-small \
        --model_id TaperFourierFormer'_ETTh2_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_origin_seq $use_origin_seq \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTh2/'0_TaperFourierFormer_'ETTh2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

fi

if false;then
# 原始序列0 + 全连接1 + relu1
use_origin_seq=0
use_linear=1
use_relu=1
# ETTh1
down_sampling_layer=7
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model TaperFourierFormer \
      --model_id TaperFourierFormer'_ETTh1_'96'_'$pred_len \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --channel_independence 1 \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_origin_seq $use_origin_seq \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
  > ../../run_log/log_07081643_local_win/ETTh1/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done
# h2
down_sampling_layer=3
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path ETT-small \
        --model_id TaperFourierFormer'_ETTh2_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTh2/'0_TaperFourierFormer_'ETTh2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#m1
down_sampling_layer=9
for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTm1_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm1/'0_TaperFourierFormer_'ETTm1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#m2
down_sampling_layer=3
for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_ETTm2_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm2/'0_TaperFourierFormer_'ETTm2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

#ecl
for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ECL_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_origin_seq $use_origin_seq \
    > ../../run_log/log_07081643_local_win/electricity/'0_TaperFourierFormer_'electricity'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done



#weather
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
        --e_layers 3 \
        --d_ff 512 \
        --d_model 512 \
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
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_origin_seq $use_origin_seq \
    > ../../run_log/log_07081643_local_win/weather/'0_TaperFourierFormer_'weather'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#exchange
down_sampling_layer=3
for pred_len in 96 192 336 720; do
 python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
      --model TaperFourierFormer \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --x_mark_len $Exchange_x_mark_len \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_x_mark_enc $use_x_mark_enc \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
  > ../../run_log/log_07081643_local_win/Exchange/'0_TaperFourierFormer_'Exchange'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done








# 原始序列0 + 全连接0 + relu0
use_origin_seq=0
use_linear=0
use_relu=0
# ETTh1
down_sampling_layer=7

for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model TaperFourierFormer \
      --model_id TaperFourierFormer'_ETTh1_'96'_'$pred_len \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --channel_independence 1 \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_origin_seq $use_origin_seq \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
  > ../../run_log/log_07081643_local_win/ETTh1/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done
# h2
down_sampling_layer=3
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path ETT-small \
        --model_id TaperFourierFormer'_ETTh2_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTh2/'0_TaperFourierFormer_'ETTh2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#m1
down_sampling_layer=9
for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTm1_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm1/'0_TaperFourierFormer_'ETTm1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#m2
down_sampling_layer=3
for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_ETTm2_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm2/'0_TaperFourierFormer_'ETTm2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

#ecl
for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ECL_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/electricity/'0_TaperFourierFormer_'electricity'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

#exchange
for pred_len in 96 192 336 720; do
 python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
      --model TaperFourierFormer \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --x_mark_len $Exchange_x_mark_len \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_x_mark_enc $use_x_mark_enc \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
  > ../../run_log/log_07081643_local_win/Exchange/'0_TaperFourierFormer_'Exchange'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done

#weather
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
        --e_layers 3 \
        --d_ff 512 \
        --d_model 512 \
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
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/weather/'0_TaperFourierFormer_'weather'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done







# 原始序列0 + 全连接1 + relu0
use_origin_seq=0
use_linear=1
use_relu=0
# ETTh1
down_sampling_layer=7

for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model TaperFourierFormer \
      --model_id TaperFourierFormer'_ETTh1_'96'_'$pred_len \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --channel_independence 1 \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_origin_seq $use_origin_seq \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
  > ../../run_log/log_07081643_local_win/ETTh1/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done
# h2
down_sampling_layer=3
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path ETT-small \
        --model_id TaperFourierFormer'_ETTh2_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTh2/'0_TaperFourierFormer_'ETTh2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#m1
down_sampling_layer=9
for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTm1_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm1/'0_TaperFourierFormer_'ETTm1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#m2
down_sampling_layer=3
for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_ETTm2_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm2/'0_TaperFourierFormer_'ETTm2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

#ecl
for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ECL_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/electricity/'0_TaperFourierFormer_'electricity'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

#exchange
for pred_len in 96 192 336 720; do
 python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
      --model TaperFourierFormer \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --x_mark_len $Exchange_x_mark_len \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_x_mark_enc $use_x_mark_enc \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
  > ../../run_log/log_07081643_local_win/Exchange/'0_TaperFourierFormer_'Exchange'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done

#weather
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
        --e_layers 3 \
        --d_ff 512 \
        --d_model 512 \
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
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/weather/'0_TaperFourierFormer_'weather'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done



# 原始序列1 + 全连接0 + relu0
use_origin_seq=1
use_linear=0
use_relu=0
# ETTh1
down_sampling_layer=7

for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model TaperFourierFormer \
      --model_id TaperFourierFormer'_ETTh1_'96'_'$pred_len \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --channel_independence 1 \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_origin_seq $use_origin_seq \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
  > ../../run_log/log_07081643_local_win/ETTh1/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done
# h2
down_sampling_layer=3
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path ETT-small \
        --model_id TaperFourierFormer'_ETTh2_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTh2/'0_TaperFourierFormer_'ETTh2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#m1
down_sampling_layer=9
for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTm1_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm1/'0_TaperFourierFormer_'ETTm1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#m2
down_sampling_layer=3
for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_ETTm2_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm2/'0_TaperFourierFormer_'ETTm2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

#ecl
for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ECL_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/electricity/'0_TaperFourierFormer_'electricity'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

#exchange
for pred_len in 96 192 336 720; do
 python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
      --model TaperFourierFormer \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --x_mark_len $Exchange_x_mark_len \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_x_mark_enc $use_x_mark_enc \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
  > ../../run_log/log_07081643_local_win/Exchange/'0_TaperFourierFormer_'Exchange'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done

#weather
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
        --e_layers 3 \
        --d_ff 512 \
        --d_model 512 \
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
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/weather/'0_TaperFourierFormer_'weather'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done







# 原始序列1 + 全连接1 + relu1
use_origin_seq=1
use_linear=1
use_relu=1
# ETTh1
down_sampling_layer=7

for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model TaperFourierFormer \
      --model_id TaperFourierFormer'_ETTh1_'96'_'$pred_len \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --channel_independence 1 \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_origin_seq $use_origin_seq \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
  > ../../run_log/log_07081643_local_win/ETTh1/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done
# h2
down_sampling_layer=3
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path ETT-small \
        --model_id TaperFourierFormer'_ETTh2_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTh2/'0_TaperFourierFormer_'ETTh2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#m1
down_sampling_layer=9
for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTm1_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm1/'0_TaperFourierFormer_'ETTm1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done
#m2
down_sampling_layer=3
for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_ETTm2_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm2/'0_TaperFourierFormer_'ETTm2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

#ecl
for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ECL_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/electricity/'0_TaperFourierFormer_'electricity'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

#exchange
for pred_len in 96 192 336 720; do
 python -u  ../../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
      --model TaperFourierFormer \
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
      --down_sampling_layers $down_sampling_layer \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --x_mark_len $Exchange_x_mark_len \
      --use_space_merge $use_space_merge \
      --use_fourier $use_fourier \
      --front_use_decomp $front_use_decomp \
      --use_conv $use_conv \
      --use_x_mark_enc $use_x_mark_enc \
      --use_linear $use_linear \
      --use_relu $use_relu \
      --use_x_mark_enc $use_x_mark_enc \
  > ../../run_log/log_07081643_local_win/Exchange/'0_TaperFourierFormer_'Exchange'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
done

#weather
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
        --e_layers 3 \
        --d_ff 512 \
        --d_model 512 \
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
        --use_linear $use_linear \
        --use_relu $use_relu \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/weather/'0_TaperFourierFormer_'weather'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_use_origin_seq='$use_origin_seq'_use_linear='$use_linear'_use_relu='$use_relu'_'0.01.log 2>&1
  done

fi


if false;then
#  h1 2分解核  dmodel=dff=256
for down_sampling_layer in 11 15 20; do

  for pred_len in 96  192 336 720; do
     python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTh1_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
      > ../../run_log/log_07081643_local_win/ETTh1/'0_TaperFourierFormer_'ETTh1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done

done



#  h2 2分解核  d_model= d_ff = 128
for down_sampling_layer in 11 15 20; do

for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --root_path ETT-small \
        --model_id TaperFourierFormer'_ETTh2_'96'_'$pred_len \
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
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTh2/'0_TaperFourierFormer_'ETTh2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done

done





# m1 d_model= d_ff = 128
for down_sampling_layer in 11 15 20; do

  for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ETTm1_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm1/'0_TaperFourierFormer_'ETTm1'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done
done



#  m2 2分解核

# m2 d_model= d_ff = 128
for down_sampling_layer in 11 15 20; do

  for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --root_path ETT-small \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_ETTm2_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --channel_independence 1 \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/ETTm2/'0_TaperFourierFormer_'ETTm2'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done
done













# electricity d_model=d_ff=512
for down_sampling_layer in 11 15 20; do

 for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_ECL_'96'_'$pred_len \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $electricity_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/electricity/'0_TaperFourierFormer_'electricity'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done
done












# Exchange d_model=d_ff=128
for down_sampling_layer in 11 15 20; do

  for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_Exchange_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $Exchange_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/Exchange/'0_TaperFourierFormer_'Exchange'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done

done










#  weather 3分解核



# weather d_ff=d_ff=512
for down_sampling_layer in 11 15 20; do

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
        --e_layers 3 \
        --d_ff 512 \
        --d_model 512 \
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
    > ../../run_log/log_07081643_local_win/weather/'0_TaperFourierFormer_'weather'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done
done









#PEMS03
for down_sampling_layer in 11 15 20; do

for pred_len in 12 24 48 96 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_PEMS03_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS03_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/PEMS03/'0_TaperFourierFormer_'PEMS03'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done
done


#PEMS04
for down_sampling_layer in 11 15 20; do

for pred_len in 12 24 48 96 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_PEMS04_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS07_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/PEMS04/'0_TaperFourierFormer_'PEMS04'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done
done

#PEMS07
for down_sampling_layer in 11 15 20; do

for pred_len in 12 24 48 96 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_PEMS07_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS07_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/PEMS07/'0_TaperFourierFormer_'PEMS07'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done
done
#PEMS08
for down_sampling_layer in 11 15 20; do

for pred_len in 12 24 48 96 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_PEMS08_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $PEMS08_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/PEMS08/'0_TaperFourierFormer_'PEMS08'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done
done
# Solar
for down_sampling_layer in 11 15 20; do

for pred_len in 96 192 336 720 ; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id TaperFourierFormer'_Solar_'96'_'$pred_len \
        --model TaperFourierFormer \
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
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $Solar_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/Solar/'0_TaperFourierFormer_'Solar'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done
done



 # Traffic d_model=d_ff=512
for down_sampling_layer in 11 15 20; do

for pred_len in 96 192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model TaperFourierFormer \
        --model_id TaperFourierFormer'_Traffic_'96'_'$pred_len \
        --root_path traffic \
        --mode regre \
        --data Traffic \
        --features M \
        --e_layers 4 \
        --d_layers 1 \
        --freq h \
        --seq_len 96 \
        --itr 1 \
        --factor 3 \
        --d_model 512 \
        --d_ff 512 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --pred_len $pred_len \
        --down_sampling_layers $down_sampling_layer \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --x_mark_len $traffic_x_mark_len \
        --use_space_merge $use_space_merge \
        --use_fourier $use_fourier \
        --front_use_decomp $front_use_decomp \
        --use_conv $use_conv \
        --use_x_mark_enc $use_x_mark_enc \
    > ../../run_log/log_07081643_local_win/Traffic/'0_TaperFourierFormer_'Traffic'_'$seq_len'_'$pred_len'_'$down_sampling_layer'_'0.01.log 2>&1
  done
done

fi