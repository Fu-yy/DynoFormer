import argparse
import os
import platform

import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_my_net import Exp_My_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    parser.add_argument('--train_only', type=bool, required=False, default=False,
                        help='perform training on full input dataset without validation and testing')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # fgn
    parser.add_argument('--feature_size', type=int, default='140', help='feature size')
    parser.add_argument('--seq_length', type=int, default=12, help='inout length')
    parser.add_argument('--pre_length', type=int, default=12, help='predict length')
    parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
    parser.add_argument('--hard_thresholding_fraction', type=int, default=1, help='hard thresholding fraction')
    parser.add_argument('--hidden_size_factor', type=int, default=1, help='hidden_size_factor')
    parser.add_argument('--sparsity_threshold', type=float, default=0.01, help='sparsity_threshold')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)


    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--device pip install Pillows', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--conv_kernel', type=int, nargs='+', default=[17,49], help='downsampling and upsampling convolution kernel_size')
    parser.add_argument('--count_params', type=int,  default=0, help='count_params')





    # TimeMixer

    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')


    # 消融
    parser.add_argument('--front_use_decomp', type=int, default=1,help='front_use_decomp')
    parser.add_argument('--use_fourier', type=int, default=1,help='use_fourier')
    parser.add_argument('--use_space_merge', type=int, default=1,help='use_space_merge')
    parser.add_argument('--use_x_mark_enc', type=int, default=1,help='use_x_mark_enc')
    parser.add_argument('--use_conv', type=int, default=1,help='use_conv')
    parser.add_argument('--use_origin_seq', type=int, default=1,help='use_origin_seq')
    parser.add_argument('--use_linear', type=int, default=1,help='use_linear')
    parser.add_argument('--use_relu', type=int, default=1,help='use_relu')



    # TaperFourierFormer
    parser.add_argument('--mode', type=str, default='regre', help='different mode of trend prediction block: [regre or mean]')
    parser.add_argument('--x_enc_len', type=int, default=7, help='x_enc_len')
    parser.add_argument('--x_mark_len', type=int, default=5, help='x_mark_len')
    # LLM
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)






    # PathFormer
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16,12,8,32,12,8,6,4,8,6,4,2])
    parser.add_argument('--num_nodes', type=int, default=21)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')

    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--metric',type=str, default='mae')

    # sageformer params
    parser.add_argument('--cls_len', type=int, default=3, help='global token length')
    parser.add_argument('--graph_depth', type=int, default=3, help='graph aggregation depth')
    parser.add_argument('--knn', type=int, default=16, help='graph nearest neighbors')
    parser.add_argument('--embed_dim', type=int, default=16, help='node embed dim')


    parser.add_argument('--time_step', type=int, default=15, help='time_step')
    parser.add_argument('--down_sampling_nums', type=int, default=3, help='down_sampling_nums')

    # 新model消融
    parser.add_argument('--use_atten', type=int, default=1, help='use_atten')
    parser.add_argument('--use_fourier_att', type=int, default=1, help='use_fourier_att')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    a = torch.cuda.is_available()
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    sys = platform.system()
    if sys == "Windows":
        print("OS is Windows!!!")
        args.num_workers = 0
    elif sys == "Linux":
        print("OS is Linux!!!")
        args.num_workers = 8

        pass
    else:
        pass



    decomp_kernel = []  # kernel of decomposition operation
    isometric_kernel = []  # kernel of isometric convolution
    for ii in args.conv_kernel:
        if ii%2 == 0:   # the kernel of decomposition operation must be odd
            decomp_kernel.append(ii+1)
            isometric_kernel.append((args.seq_len + args.pred_len+ii) // ii)
        else:
            decomp_kernel.append(ii)
            isometric_kernel.append((args.seq_len + args.pred_len+ii-1) // ii)
    args.isometric_kernel = isometric_kernel  # kernel of isometric convolution
    args.decomp_kernel = decomp_kernel   # kernel of decomposition operation





    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    elif args.task_name == 'my_net':
        Exp = Exp_My_Forecast

    else:
        Exp = Exp_Long_Term_Forecast



    data_parser = {
        'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'WTH':{'data':'weather.csv','T':'OT','M':[21,21,21],'S':[1,1,1],'MS':[21,21,1]},
        'ECL':{'data':'electricity.csv','T':'OT','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
        'Traffic': {'data': 'traffic.csv', 'T': 'OT', 'M': [862, 862, 862], 'S': [1, 1, 1], 'MS': [862, 862, 1]},
        'Exchange': {'data': 'exchange_rate.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1]},
        'Solar': {'data': 'solar_AL.txt', 'T': 'OT', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
        'ILI': {'data': 'national_illness.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'PEMS03': {'data': 'PEMS03.npz', 'T': 'OT', 'M': [358, 358, 358], 'S': [1, 1, 1], 'MS': [358, 358, 1]},
        'PEMS04': {'data': 'PEMS04.npz', 'T': 'OT', 'M': [307, 307, 307], 'S': [1, 1, 1], 'MS': [307, 307, 1]},
        'PEMS07': {'data': 'PEMS07.npz', 'T': 'OT', 'M': [883, 883, 883], 'S': [1, 1, 1], 'MS': [883, 883, 1]},
        'PEMS08': {'data': 'PEMS08.npz', 'T': 'OT', 'M': [170, 170, 170], 'S': [1, 1, 1], 'MS': [170, 170, 1]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    print('Args in experiment:')
    print(args)
    print_args(args)




    if args.count_params == 1:
        exp = Exp(args)  # set experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_lr{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_d_model{}_k{}_layer_nums{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.learning_rate,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.d_model,
            args.k,
            args.layer_nums,
            # args.down_sampling_layers,
            args.des

        )

        print('>>>>>>>count params : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.train_count_params(setting)


    # args.batch_size = 7
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_lr{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_d_model{}_k{}_layer_nums{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.learning_rate,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.d_model,
                args.k,
                args.layer_nums,
                # args.down_sampling_layers,
                args.des,
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_lr{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_d_model{}_k{}_layer_nums{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.learning_rate,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.d_model,
            args.k,
            args.layer_nums,
            # args.down_sampling_layers,
            args.des,
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()



