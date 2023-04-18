from retro_star.api import RSPlanner
from retro_star.common import args
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import sys
import torch
import json, os
import logging
import signal
import time, random

# add the following lines of code to the end of retro_star/common/parse_args.py
# parser.add_argument('--input', type=str, default='test100.smi')
# parser.add_argument('--output', type=str, default=None)
# parser.add_argument('--left', type=int, default=0)
# parser.add_argument('--right', type=int, default=2000)
# parser.add_argument('--num_threads', type=int, default=1)

torch.set_num_threads(1)
# logging.disable()
planner = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=100,
    expansion_topk=50
)

if __name__ == '__main__':
    # args = get_args()
    # args = sys.argv
    # print(args)
    # input()
    print(f'Input: {args.input}')
    with open(args.input) as f:
        lines = f.readlines()
    print(f'Input entries: {len(lines)}')
    # cids, smiles = [], []
    # for line in lines:
    #     cid, smile = line.split()
    #     cids.append(cid)
    #     smiles.append(smile)
    # Parallel(n_jobs=16, verbose=10)(delayed(worker)(cids[i], smiles[i]) for i in range(len(smiles)))
    # print(len(results))
    results = {}
    correct = 0
    save_results = {}
    if args.output is None:
        output = args.input.split('.')[0] + '_synth_results.txt'
    else:
        output = args.output
    with open(output, 'w') as f:
        for line in tqdm(lines[args.left: args.right], dynamic_ncols=True):
            cid, smile = line.split()
            try:
                # signal.alarm(1200)
                res = planner.plan(smile)
                # signal.alarm(0)
            except:
                res = None
            if res is not None:
                results[cid] = 1
                correct += 1
            else:
                results[cid] = 0
            save_results[cid] = res
            f.write(f'{cid} {results[cid]}\n')
    # with open(args.output, 'w') as f:
    #     for k, v in results.items():
    #         f.write(f'{k} {v}\n')
    print(f'{correct} / {min(len(lines), args.right - args.left)} synthesizable.')
    with open(f'{output.split(".")[0]}.json', 'w') as f:
        json.dump(save_results, f)
    print(f'save results to {output.split(".")[0]}.json')
    
# only cpu, 100 molecules, 17:47
# gpu, 100 molecules, 16:36
# gpu, 8 threads, 100 molecules, 16:05
# gpu, 4 threads, 100 molecules, 16:03
# gpu, 2 threads, 100 molecules, 15:33
# gpu, 1 threads, 100 molecules,