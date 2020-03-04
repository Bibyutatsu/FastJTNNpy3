import torch
import torch.nn as nn

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit


def sample(vocab, output_file, hidden_size, latent_size, depthT, depthG, model, nsample):
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model.load_state_dict(torch.load(model))
    model = model.cuda()

    torch.manual_seed(0)
    with open(output_file, 'w') as out_file:
        for i in range(nsample):
            out_file.write(str(model.sample_prior())+'\n')

if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsample', type=int, required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    args = parser.parse_args()
    
    sample(args.vocab, args.output_file, args.hidden_size, args.latent_size, args.depthT, args.depthG, args.model, args.nsample)