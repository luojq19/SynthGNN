# GNN for synthetic accessibility predictions

### Prepare Dataset
Run `python datasets.py` to generate the preprocessed dataset to save time in later experiements.

Current dataset is 340k molecules randomly sampled from PubChem, and run Retro* to label their synthetic accessiblity. During training we balanced the positive and negative molecules, resulting in about 110k training data.

### Download Checkpoints
download the `log` directory from [here](https://drive.google.com/file/d/1FTEBtXyHd5vecNkeJEIbEtSWm0wIZ68W/view?usp=sharing), unzip and put it in the current directory.

Currently the best performing model is `exp_GATv2_4layer-20230418-0540` with 87% test accuracy. Refer to its corresponding subdirectory under `log` for detailed model architecture and training configurations.