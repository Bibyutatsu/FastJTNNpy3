import torch
import torch.nn as nn
from multiprocessing import Pool
import numpy as np

import math, random, sys
from optparse import OptionParser
import pickle

from fast_jtnn import *
import rdkit

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    with open(opts.train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    all_data = pool.map(tensorize, data)
    all_data_split = np.array_split(all_data, num_splits)

    for split_id in range(num_splits):
        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(all_data_split[split_id], f)

