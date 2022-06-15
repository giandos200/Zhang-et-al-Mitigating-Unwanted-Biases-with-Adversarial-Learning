# Zhang-et-al-mitigating-unwanted-biases-with-adversarial-learning

Reproduction of Zhang, et al. ["Learning Fair Representations"](https://dl.acm.org/doi/pdf/10.1145/3278721.3278779), AAAI 2018.

The code take inspiration from two repositories.
- The author paper: https://dl.acm.org/doi/pdf/10.1145/3278721.3278779
- AIF360 implementation: https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/inprocessing/adversarial_debiasing.py

....(Motivations)....


### Installation guidelines:

Following, the instruction to install the correct packages for running the experiments.
run:

```bash
#sudo chmod +x create_venv.sh + User_password if Permission Denied 
./create_venv.sh
```

or manually:

```bash
python3 -m venv venv_adv
source venv_adv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Training and test the model

To train and evaluate all the dataset on a Deep Neural Network with/without Adversarial Debiasing through Accuracy, 
Difference in Equal Opportunity, and Difference in Average Odds, run the following command:

```bash
#sudo chmod +x trainAll.sh + User_password if Permission Denied 
./trainAll.sh
```

To run a single dataset, execute `python3 main.py {dataset}_{sensitive}_{Backend}` or chose one of this:

```
python main.py adult_gender_TF2
python main.py adult_gender_Torch
python main.py adult_marital-status_TF2
python main.py adult_marital-status_Torch
python main.py german_gender_TF2
python main.py german_gender_Torch
python main.py german_foreignworker_TF2
python main.py german_foreignworker_Torch
```

### Results