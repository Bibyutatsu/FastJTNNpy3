#######################################
# Bayesian Optimization

For Bayesian optimization, we used the scripts from https://github.com/mkusner/grammarVAE

This requires you to install their customized Theano library.
Please see https://github.com/mkusner/grammarVAE#bayesian-optimization for installation.

## Usage
First generate the latent representation of all training molecules:
```
python gen_latent.py --data ../data/train.txt --vocab ../data/vocab.txt --model ../fast_molvae/vae_model/model.epoch-19 --output './descriptors/'
```
This generates `latent_features.txt` for latent vectors and other files for logP, synthetic accessability scores.

To run Bayesian optimization:

```
python run_bo.py --vocab ../data/vocab.txt --save_dir results --seed 1 --model ../fast_molvae/vae_model/model.epoch-19 --descriptors './descriptors/'
```
It performs two iterations of Bayesian optimization with EI heuristics, and saves discovered molecules in `results/`
Following previous work, we tried `seed = 1`.