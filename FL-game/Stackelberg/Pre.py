
from math import floor
from matplotlib import scale
import numpy as np
import argparse
import importlib
import torch
import os
import sys
from src.utils.worker_utils import read_data
from src.utils.client_utils import generate_real_time_v1, generate_time
from src.utils.config_utils import NAME_MAP
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS
import argparse
# from src.controllers.server import FedIoTServer
from src.controllers.client import FedIoTClient 
import src.communication.comm_config as connConfig
def getA(Log1, Log2):
    ret = []
    stop_round = [19, 29, 49]

    for j in range(10):
        for i in range(3):
            R = stop_round[i]
            alpha = (R+1)*(Log1[j]['loss'][R] - Log2[j]['loss'][R])
            A1, A2 = 0, 0
            pk1 = Log1[j]['pk']
            pk2 = Log2[j]['pk']
            qk1 = Log1[j]['qk']
            qk2 = Log2[j]['qk']
            Gk1 = Log1[j]['gk'][R]
            Gk2 = Log2[j]['gk'][R]

            for k in range(len(pk1)):
                A1 += (1-qk1[k])/qk1[k] * (pk1[k]* Gk1[k])**2
                A2 += (1-qk2[k])/qk2[k] * (pk2[k]* Gk2[k])**2
            if A1 == A2:
                print("There is an zero", A1, A2)
            alpha /= (A1 - A2)
            ret.append(alpha)

    avg_ret = []

    for i in range(3):
        R = stop_round[i]
        alpha = 0
        for j in range(10):
            alpha += (R+1)*(Log1[j]['loss'][R] - Log2[j]['loss'][R])
        alpha /= 10
        A1, A2 = 0, 0
        pk1 = Log1[0]['pk']
        pk2 = Log2[0]['pk']
        qk1 = Log1[0]['qk']
        qk2 = Log2[0]['qk']
        for j in range(10):
            Gk1 = Log1[j]['gk'][R]
            Gk2 = Log2[j]['gk'][R]
            for k in range(len(pk1)):
                A1 += (1-qk1[k])/qk1[k] * (pk1[k]* Gk1[k])**2
                A2 += (1-qk2[k])/qk2[k] * (pk2[k]* Gk2[k])**2
        A1 /= 10
        A2 /= 10

        if A1 == A2:
            print("There is an zero")
        alpha /= (A1 - A2)
        avg_ret.append(alpha)
    
    return ret, avg_ret

def fl_experiment(args, fedServer):
    # config settings 
    fedServer.config_experiment(args)
    
    # transfer config info to clients
    fedServer.init_clients()

    # deploy data
    # fedServer.deploy_data()

    # test speed of each client
    fedServer.test_comm_speed()
    
    fedServer.deploy_data()
    fedServer.train()
    # fedServer.wait()
    return fedServer.trainer.logDict

def client_main():
    fedClient = FedIoTClient()
    fedClient.connect2server()
    while fedClient.is_experiment_ongoing():
        fedClient.init_config()
        fedClient.config_experiment()
        fedClient.test_comm_speed()
        fedClient.deploy_data()
        fedClient.train()
    
    print("Experiment Ended.")
def server_main():
    # Modify parameters here

   
    SEED_MAX = 1
    

    args = {
        'num_epoch': 50,
        'batch_size': 24,
        'num_round': 50,
        'model': 'logistic',
        'update_rate': 0,
        'num_clients': 40,
        'dataset': 'mnist_niid1_7_0',
        'lr': 0.1,
        'wd': 0.001,
        'gpu': False,
        'noaverage': False,
        'experiment_folder': 'bench',
        'is_sys_heter': True,
        'without_rp': False, ##
        'decay': 'soft', ## API ## 
        'test_num':2,
        'C': [5.0] * N, # cost
        'budget': 15.0,
        'v': np.random.exponential(scale=100,size=40), # intrinsic value
        'optim_method':'matlab',

    }
    
    N = int(args['num_clients'])
    fedServer = FedIoTServer(args)
    fedServer.connect2clients()

    experimentConfig = {
        'seedMax': SEED_MAX,
    }
    Log1 = []
    Log2 = []
    for t_seed in range(0, 1):
        args['time_seed'] = t_seed
        for seed in range(1, 1+SEED_MAX):
            args['seed'] = seed

            args['qk'] = 0.1
            args['algo'] = NAME_MAP['pretrain']
            fedServer = FedIoTServer(args)
            fedServer.connect2clients()
            fedServer.start_experiment()
            log = fl_experiment(args, fedServer)
            Log1.append(log)
            fedServer.end_experiment()


            args['qk'] = 0.3
            args['algo'] = NAME_MAP['pretrain']
            fedServer = FedIoTServer(args)
            fedServer.connect2clients()
            fedServer.start_experiment()
            log = fl_experiment(args, fedServer)
            Log2.append(log)
            fedServer.end_experiment()


            ret, avg_ret = getA(Log1, Log2)
        print(ret, avg_ret)

    

    print("Experiment Ended.")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='server or client')
    args = parser.parse_args()
            
    if args.mode == 'server':
        from src.controllers.server import FedIoTServer
        server_main()
    elif args.mode == 'client':
        client_main()
    else:
        raise Exception("Wrong parser parameter!")
        
        