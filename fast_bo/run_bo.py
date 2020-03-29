import sys
sys.path.append('../')
import pickle
import gzip
import scipy.stats as sps
import numpy as np
import os
import rdkit
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors
import torch
import torch.nn as nn
from fast_jtnn import create_var, JTNNVAE, Vocab, sascorer
from fast_jtnn.sparse_gp import SparseGP
import networkx as nx
from rdkit.Chem import rdmolops
from tqdm import tqdm
from optparse import OptionParser
import joblib


# We define the functions used to load and save objects
def save_object(obj, filename):
    joblib.dump(obj, filename)


def load_object(filename):
    return joblib.load(filename)


def save_object_old(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest:
        dest.write(result)
    dest.close()


def load_object_old(filename):
    with gzip.GzipFile(filename, 'rb') as source:
        result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret


def main_bo(vocab_path,
            model_path,
            save_dir,
            descriptor_path,
            sampling=60,
            iterations=2,
            epochs=2,
            hidden_size=450,
            latent_size=56,
            depthT=20,
            depthG=3,
            random_seed=1):
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)

    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    # We load the random seed
    np.random.seed(random_seed)
    
    # Path of the files
    latent_feature = os.path.join(descriptor_path, './latent_features.txt')
    target = os.path.join(descriptor_path, './targets.txt')
    logp_value = os.path.join(descriptor_path, './logP_values.txt')
    sa_score = os.path.join(descriptor_path, './SA_scores.txt')
    cycle_score = os.path.join(descriptor_path, './cycle_scores.txt')
    
    # We load the data (y is minued!)
    X = np.loadtxt(latent_feature)
    y = -np.loadtxt(target)
    y = y.reshape((-1, 1))

    n = X.shape[0]
    permutation = np.random.choice(n, n, replace=False)

    X_train = X[permutation, :][0:np.int(np.round(0.9 * n)), :]
    X_test = X[permutation, :][np.int(np.round(0.9 * n)):, :]

    y_train = y[permutation][0: np.int(np.round(0.9 * n))]
    y_test = y[permutation][np.int(np.round(0.9 * n)):]

    np.random.seed(random_seed)

    logP_values = np.loadtxt(logp_value)
    SA_scores = np.loadtxt(sa_score)
    cycle_scores = np.loadtxt(cycle_score)

    iteration = 0
    while iteration < iterations:
        # We fit the GP
        np.random.seed(iteration * random_seed)
        M = 500
        sgp = SparseGP(X_train, 0 * X_train, y_train, M)
        sgp.train_via_ADAM(X_train,
                           0 * X_train,
                           y_train,
                           X_test,
                           X_test * 0,
                           y_test,
                           minibatch_size=10 * M,
                           max_iterations=5,
                           learning_rate=0.001)

        pred, uncert = sgp.predict(X_test, 0 * X_test)
        error = np.sqrt(np.mean((pred - y_test)**2))
        testll = np.mean(sps.norm.logpdf(pred - y_test, scale=np.sqrt(uncert)))
        print('Test RMSE: ', error)
        print('Test ll: ', testll)

        pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(
            sps.norm.logpdf(pred - y_train, scale=np.sqrt(uncert)))
        print('Train RMSE: ', error)
        print('Train ll: ', trainll)

        # We pick the next 60 inputs
        next_inputs = sgp.batched_greedy_ei(sampling,
                                            np.min(X_train, 0),
                                            np.max(X_train, 0))
        # joblib.dump(next_inputs, './next_inputs.pkl')
        # next_inputs = joblib.load('./next_inputs.pkl')
        valid_smiles = []
        new_features = []
        for i in tqdm(range(sampling)):
            all_vec = next_inputs[i].reshape((1, -1))
            tree_vec, mol_vec = np.hsplit(all_vec, 2)
            tree_vec = create_var(torch.from_numpy(tree_vec).float())
            mol_vec = create_var(torch.from_numpy(mol_vec).float())
            tree_vecs, _ = model.rsample(tree_vec, model.T_mean, model.T_var)
            mol_vecs, _ = model.rsample(mol_vec, model.G_mean, model.G_var)
            s = model.decode(tree_vecs, mol_vecs, prob_decode=False)
            if s is not None:
                valid_smiles.append(s)
                new_features.append(all_vec)

        print(len(valid_smiles), "molecules are found")
        valid_smiles = valid_smiles
        new_features = next_inputs
        new_features = np.vstack(new_features)
        save_object(
            valid_smiles,
            os.path.join(save_dir, "valid_smiles{}.pkl".format(iteration))
        )

        scores = []
        for i in range(len(valid_smiles)):
            current_log_P_value = Descriptors.MolLogP(
                MolFromSmiles(valid_smiles[i]))
            current_SA_score = -sascorer.calculateScore(
                MolFromSmiles(valid_smiles[i]))
            cycle_list = nx.cycle_basis(
                nx.Graph(rdmolops.GetAdjacencyMatrix(
                    MolFromSmiles(valid_smiles[i]))))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6

            current_cycle_score = -cycle_length

            current_SA_score_normalized = (
                current_SA_score - np.mean(
                    SA_scores)) / np.std(SA_scores)

            current_log_P_value_normalized = (
                current_log_P_value - np.mean(
                    logP_values)) / np.std(logP_values)

            current_cycle_score_normalized = (
                current_cycle_score - np.mean(
                    cycle_scores)) / np.std(cycle_scores)

            score = (current_SA_score_normalized +
                     current_log_P_value_normalized +
                     current_cycle_score_normalized)

            scores.append(-score)  # target is always minused

        print(valid_smiles)
        print(scores)

        save_object(
            scores,
            os.path.join(save_dir, "scores{}.pkl".format(iteration))
        )

        if len(new_features) > 0:
            X_train = np.concatenate([X_train, new_features], 0)
            y_train = np.concatenate([y_train, np.array(scores)[:, None]], 0)

        iteration += 1


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-v", "--vocab", dest="vocab_path")
    parser.add_option("-m", "--model", dest="model_path")
    parser.add_option("-o", "--save_dir", dest="save_dir")
    parser.add_option("-f", "--descriptors", dest="descriptor_path")
    parser.add_option("-b", "--sampling", dest="sampling", default=60)
    parser.add_option("-i", "--iteration", dest="iteration", default=2)
    parser.add_option("-e", "--epochs", dest="epochs", default=2)
    parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
    parser.add_option("-l", "--latent", dest="latent_size", default=56)
    parser.add_option("-t", "--depthT", dest="depthT", default=20)
    parser.add_option("-g", "--depthG", dest="depthG", default=3)
    parser.add_option("-r", "--seed", dest="random_seed", default=1)
    opts, args = parser.parse_args()

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    depthT = int(opts.depthT)
    depthG = int(opts.depthG)
    random_seed = int(opts.random_seed)
    iteration = int(opts.iteration)
    epochs = int(opts.epochs)
    sampling = int(opts.sampling)

    main_bo(opts.vocab_path,
            opts.model_path,
            opts.save_dir,
            opts.descriptor_path,
            sampling,
            iteration,
            epochs,
            hidden_size,
            latent_size,
            depthT,
            depthG,
            random_seed)
