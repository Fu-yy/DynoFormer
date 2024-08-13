source activate fuypy310t2cu118


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
if [ ! -d "../../run_log/log_07201154_win" ]; then
    mkdir ../../run_log/log_07201154_win
fi
if [ ! -d "../../run_log/log_07201154_win/ETTm1" ]; then
    mkdir ../../run_log/log_07201154_win/ETTm1
fi
if [ ! -d "../../run_log/log_07201154_win/ETTh1" ]; then
    mkdir ../../run_log/log_07201154_win/ETTh1
fi
if [ ! -d "../../run_log/log_07201154_win/ETTm2" ]; then
    mkdir ../../run_log/log_07201154_win/ETTm2
fi

if [ ! -d "../../run_log/log_07201154_win/ETTh2" ]; then
    mkdir ../../run_log/log_07201154_win/ETTh2
fi


if [ ! -d "../../run_log/log_07201154_win/Exchange" ]; then
    mkdir ../../run_log/log_07201154_win/Exchange
fi


if [ ! -d "../../run_log/log_07201154_win/weather" ]; then
    mkdir ../../run_log/log_07201154_win/weather
fi





layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=5
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model DynoFormer \
        --model_id DynoFormer'_ETTh1_'96'_'$pred_len \
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
        > ../../run_log/log_07201154_win/ETTh1/'0_DynoFormer_'ETTh1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done


#  h2 ok
layer_nums=3
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=4
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model DynoFormer \
        --root_path ETT-small \
        --model_id DynoFormer'_ETTh2_'96'_'$pred_len \
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
        > ../../run_log/log_07201154_win/ETTh2/'0_DynoFormer_'ETTh2'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
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
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model DynoFormer \
        --root_path ETT-small \
        --model_id DynoFormer'_ETTm1_'96'_'$pred_len \
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
        --k $k \
        > ../../run_log/log_07201154_win/ETTm1/'0_DynoFormer_'ETTm1'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
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
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model DynoFormer \
        --root_path ETT-small \
        --model_id DynoFormer'_ETTm2_'96'_'$pred_len \
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
        > ../../run_log/log_07201154_win/ETTm2/'0_DynoFormer_'ETTm2'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done






# Exchange d_model=d_ff=128  ok

layer_nums=2
d_model=128
d_ff=128
down_sampling_layers=3
use_origin_seq=1
k=5
learning_rate=0.0005

for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model DynoFormer \
        --root_path exchange_rate \
        --model_id DynoFormer'_Exchange_'96'_'$pred_len \
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
        > ../../run_log/log_07201154_win/Exchange/'0_DynoFormer_'Exchange'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done


#weather
layer_nums=2
d_model=64
d_ff=64
down_sampling_layers=3
use_origin_seq=1
k=4
for pred_len in 96  192 336 720; do
   python -u  ../../run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model DynoFormer \
        --root_path weather \
        --model_id DynoFormer'_WTH_'96'_'$pred_len \
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
        > ../../run_log/log_07201154_win/weather/'0_DynoFormer_'weather'_'$seq_len'_'$pred_len'_layer_nums='$layer_nums'_d_model='$d_model'_k='$k'_use_origin_seq='$use_origin_seq'down_sampling_layers='$down_sampling_layers'_'0.01.log 2>&1
  done

