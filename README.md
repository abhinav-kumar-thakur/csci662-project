# csci662-project
# PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction
The work in this project focuses on the reproduction of the claims in paper:
**PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction. Hengyi Zheng, Rui Wen, Xi Chen et al. ACL 2021.**
* Link: [https://aclanthology.org/2021.acl-long.486/](https://aclanthology.org/2021.acl-long.486/)
* Source code: [https://github.com/hy-struggle/PRGC](https://github.com/hy-struggle/PRGC)

## Reproduction using source code shared by authors
While evaluating the model for overlapping examples, we faced an issue with evaluation scripts share by the author.
The updated code can be found at [https://github.com/abhinav-kumar-thakur/PRGC](https://github.com/abhinav-kumar-thakur/PRGC) and changes can be seen in the [commit](https://github.com/abhinav-kumar-thakur/PRGC/commit/5a9c77a438ae59076972dc65333d1b78e66eac07#diff-3972a695ff852620b980ed5846b894f4d4e00adb09d8abfc9a81c53564dc1b56).

* To setup and run the PRGC source code:
```shell
git clone https://github.com/abhinav-kumar-thakur/csci662-project.git
cd csci662-project
pip install torch==1.10.0 transformers==3.2.0 tqdm pandas ptflops
sh download_and_setup_prgc.sh
# Update parameters below for trying training with different hyperparameters
python train.py --ex_index=1 --epoch_num=100 --device_id=0 --corpus_type=WebNLG --ensure_corres --ensure_rel --rel_threshold 0.1 --corres_threshold 0.5
```
* Update `prgc_reproduction.sh` to try out various hyper-parameters
* To verify computational complexity (not including embedding models) run `python3 prgc_computation_analysis.py`
* To create evaluation data for overlapping patterns, run: `python3 data/process4detailed.py` from cloned PRGC directory
* To evaluate for a particular overlapping pattern, run: `sh ./script/evaluate.sh <PATTEN_TYPE>` from cloned PRGC directory
* Supported pattern types are: `SEO, EPO, SSO`.

# Implementation from scratch
## Setup 
Python version: 3.7.9
### Using conda
```shell
conda create -n myprgc-env -c pytorch -c huggingface python=3.7.9 pytorch=1.10.0 transformers=3.2.0 tqdm pandas ptflops
conda activate myprgc-env
```
### Using pip
```shell
# To create environment
python3 -m venv myprgc-env
source myprgc-env/bin/activate
# To install dependencies
pip install torch==1.10.0 transformers==3.2.0 tqdm pandas ptflops
```
### Downloading pretraing bert-base-uncased model
```shell
wget https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin
mv pytorch_model.bin ./pretrained_model
```

## Training
Training can be triggered using train.py script, please find the usage below:

```
usage: train.py [-h] [-dataset DATASET] [-checkpoint CHECKPOINT] [-nepochs NEPOCHS] [-batchsize BATCHSIZE] [-lambda1 LAMBDA1] [-lambda2 LAMBDA2] [-gpuid GPUID] [-seed SEED] [-fusion FUSION] [-opt OPT]

PRGC Model

optional arguments:
  -h, --help            show this help message and exit
  -dataset DATASET      Dataset Choice out of {'NYT','NYT-star','WebNLG','WebNLG-star'}
  -checkpoint CHECKPOINT
                        chepoint for a pre-trained language model, from https://huggingface.co/models
  -nepochs NEPOCHS      number of training epochs
  -batchsize BATCHSIZE  size of each batch
  -lambda1 LAMBDA1      threshold for relation judgement, in [0,1]
  -lambda2 LAMBDA2      threshold for global correspondence, in [0,1]
  -gpuid GPUID          GPU id
  -seed SEED            RNG seed
  -fusion FUSION        Fusion type concat or sum
  -opt OPT              optimizer from {'bertadam','adam'}
```

# Colab Demo Page
**For help refer to [Colab page](https://colab.research.google.com/drive/1K0dLh1dv779k1tZbjgrbKM9R2_Ck9yNx?usp=sharing):**
It guides through the execution of:
* PRGC model using author's source code
* PRGC model implemented from scratch

