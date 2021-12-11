# csci662-project
# PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction
The work in this project focuses on the reproduction of the claims in paper:
**PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction. Hengyi Zheng, Rui Wen, Xi Chen et al. ACL 2021.**
* Link: [https://aclanthology.org/2021.acl-long.486/](https://aclanthology.org/2021.acl-long.486/)
* Source code: [https://github.com/hy-struggle/PRGC](https://github.com/hy-struggle/PRGC)
* To reproduce run `sh ./prgc_reproduction.sh`
* To verify computational complexity (not including embedding models)

# Implementation from scratch
## Setup 
Python version: 3.7.9
### Using conda
```shell
conda create -n myprgc-env -c pytorch -c huggingface python=3.7.9 pytorch=1.6.0 transformers=3.2.0 tqdm pandas
```
### Using pip
```shell
python3 -m venv myprgc-env
source myprgc-env/bin/activate
pip3 install torch transformers tqdm pandas
```
### Downloading pretraing bert-base-uncased model
```shell
wget https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin
mv pytorch_model.bin ./pretrained_model
```

## Training
Run:
```shell
python3 prgc.py -dataset 'WebNLG' -checkpoint 'bert-base-uncased' -nepochs 100 -batchsize 6 -lambda1 0.1 -lambda2 0.5 -seed 100
```
