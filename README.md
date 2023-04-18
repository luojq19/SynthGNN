# GNN for synthetic accessibility predictions

### Prepare Dataset
Run `python datasets.py` to generate the preprocessed dataset to save time in later experiements.

Current dataset is 340k molecules randomly sampled from PubChem, and run Retro* to label their synthetic accessiblity. During training we balanced the positive and negative molecules, resulting in about 110k training data.

### Download Checkpoints
download the `log` directory from [here]() and put it in the current directory.