# FastJTNNpy3 : Junction Tree Variational Autoencoder for Molecular Graph Generation
Python 3 Version of Fast Junction Tree Variational Autoencoder for Molecular Graph Generation (ICML 2018)

<img src="https://github.com/Bibyutatsu/FastJTNNpy3/blob/master/Old/paradigm.png" width="600">

Implementation of our Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)

# Requirements
* RDKit (version >= 2017.09)    : Tested on 2019.09.1
* Python (version >= 3.6)       : Tested on 3.7.4
* PyTorch (version >= 0.2)      : Tested on 1.0.1

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.

# Quick Start

## Code for Accelerated Training
This repository contains the Python 3 implementation of the new Fast Junction Tree Variational Autoencoder code.

* `fast_molvae/` contains codes for VAE training. Please refer to `fast_molvae/README.md` for details.
* `fast_molvae/fast_jtnn/` contains codes for model implementation.

## Old codes
This repository contains the following directories:

* `Old/bo` includes scripts for Bayesian optimization experiments. Please read `Old/bo/README.md` for details.
* `Old/molvae/` includes scripts for training our VAE model only. Please read `Old/molvae/README.md` for training our VAE model.
* `Old/molopt/` includes scripts for jointly training our VAE and property predictors. Please read `Old/molopt/README.md` for details.
* `Old/molvae/jtnn/` contains codes for model formulation. Please read `Old/molvae/README.md` for training our VAE model.

# Contact
Bibhash Chandra Mitra (bibhashm220896@gmail.com)
