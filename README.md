# Natural Language Inference Enhanced by Knowledge Graph Embedding (KEIM)

# Dependencies
To run it perfectly, you will need (recommend using Ananconda to set up environment):
* Python 3.5 or 3.6
* Tensorflow 1.10.0
* Java >= 1.6

# 1. Download and preprocess 

```Bash
cd data_process
```

Download the data and the resources for preprocess
```Python
python download.py
```
Preprocess for SNLI dataset
[2] python preprocess_data_snli.py

Preprocess for MultiNLI dataset
[3] python preprocess_data_multinli.py

Preprocess for SciTail dataset
[4] python process_data_scitail.py

# 2. Train KEIM
Hyper-parameters are set in configure file in ./config/xxx.sample.config

cd src

Training process for SNLI dataset
python Main.py --config_path ../configs/snli.sample.config

Training process for MultiNLI dataset
python Main.py --config_path ../configs/multinli.sample.config

Training process for SciTail dataset
python Main.py --config_path ../configs/scitail.sample.config

The model and results are saved in $model_dir$.

# 3. Evaluation

cd src

python Evaluation.py --model_prefix your_model --in_path The path to the test file.
