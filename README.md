# Zhang-et-al-mitigating-unwanted-biases-with-adversarial-learning

Reproduction of Zhang, et al. ["Mitigating Unwanted Biases with Adversarial Learning"](https://dl.acm.org/doi/pdf/10.1145/3278721.3278779), AAAI 2018.

The code take inspiration from two repositories.
- The author paper: https://dl.acm.org/doi/pdf/10.1145/3278721.3278779
- AIF360 implementation: https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/inprocessing/adversarial_debiasing.py

....(Motivations)....Soon.....


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
Following, the Results for each dataset and sensitive feature cofiguration.

The metrics used to measure the Fairness-Accuracy trade-off performance are:

- <img src="https://latex.codecogs.com/svg.image?ACC&space;=&space;\frac{TP&plus;TN}{TP&plus;TN&plus;FP&plus;FN}" title="ACC = \frac{TP+TN}{TP+TN+FP+FN}" />
- <img src="https://latex.codecogs.com/svg.image?DEO&space;=&space;\mid&space;TPrate_{privileged}&space;-&space;TPrate_{unprivileged}&space;\mid" title="DEO = \mid TPrate_{privileged} - TPrate_{unprivileged} \mid" />
- <img src="https://latex.codecogs.com/svg.image?DAO&space;=&space;\frac{\mid&space;FPrate_{privileged}-&space;FPrate_{unprivileged}\mid&plus;\mid&space;TPrate_{privileged}-TPrate_{unprivileged}\mid}{2}" title="DAO = \frac{\mid FPrate_{privileged}- FPrate_{unprivileged}\mid+\mid TPrate_{privileged}-TPrate_{unprivileged}\mid}{2}" />

**Adult dataset with *gender* sensitive feature** 

| Backend | Debiased | Dataset | Accuracy | DEO                 | DAO                  | Train/Inference Time (sec) |
| ------- | -------- | ------- | -------- |---------------------|----------------------|----------------------------|
| TF2     | False    | Train   | 0.8642227081746481  | 0.13559745161364456 | 0.08697951237072674  | 84.03344511985779          |
| TF2     | False    | Test    | 0.854742427592306   | 0.12976062354191595 | 0.09005127108163935  | 0.024374723434448242       |
| TF2     | True     | Train   | 0.8622324872846999  | 0.10353353361205019 | 0.05532559740620209  | 346.81407475471497         |
| TF2     | True     | Test    | 0.849878399292505   | 0.09861873344162428 | 0.06049675684227179  | 0.02435469627380371        |
| Torch   | False    | Train   | 0.8729452812108406  | 0.14593587916012846 | 0.09511375800843087  | 41.022071838378906         |
| Torch   | False    | Test    | 0.8518682290515145  | 0.1375462074774227  | 0.09946199901513178  | 0.0319821834564209         |
| Torch   | True     | Train   | 0.7348583503280179  | 0.07205142844797205 | 0.11272524371899716  | 92.79310917854309          |
| Torch   | True     | Test    | 0.7357948264426266  | 0.07263737463540935 |  0.11443645260265914 | 0.024663925170898438       |


**Adult dataset with *Marital Status* sensitive feature** 

| Backend | Debiased | Dataset | Accuracy            | DEO                  | DAO                  | Train/Inference Time (sec) |
| ------- | -------- | ------- |---------------------|----------------------|----------------------|----------------------------|
| TF2     | False    | Train   | 0.8635101599547901  | 0.27850694614133636  | 0.18347017835792262  | 104.02185273170471         |
| TF2     | False    | Test    | 0.8611541012602255  | 0.2794416553898665   | 0.1845815647627876   | 0.030504703521728516       |
| TF2     | True     | Train   | 0.8356224968672449  | 0.16459003951735313  | 0.08795286006057038  | 357.73937916755676         |
| TF2     | True     | Test    | 0.8257793499889454  | 0.17715200134146256  | 0.09953424921584775  | 0.027960777282714844       |
| Torch   | False    | Train   | 0.8695299638811764  | 0.28129974688958703  | 0.18248370459580962  | 44.639039278030396         |
| Torch   | False    | Test    | 0.861375193455671   | 0.28448663591436896  | 0.1858514291042525   | 0.026957035064697266       |
| Torch   | True     | Train   | 0.7377822550922627  | 0.024332888821577723 | 0.020144712285895247 | 99.74486684799194          |
| Torch   | True     | Test    | 0.7300464293610436  | 0.026908391114691132 | 0.025953421787326472 | 0.025796890258789062       |


**German dataset with *gender* sensitive feature** 

| Backend | Debiased | Dataset | Accuracy           | DEO                  | DAO                  | Train/Inference Time (sec) |
|---------| -------- | ------- |--------------------|----------------------|----------------------|----------------------------|
| TF2     | False    | Train   | 0.7788888888888889 | 0.06295776842761414  | 0.03794608072307931  | 11.139900922775269         |
| TF2     | False    | Test    | 0.76               | 0.2150537634408602   | 0.14633006077606356  | 0.007011890411376953       |
| TF2     | True     | Train   | 0.8044444444444444 | 0.060256610046231285 | 0.04088099319515866  | 34.61143755912781          |
| TF2     | True     | Test    | 0.79               | 0.22954651706404866  | 0.1680691912108462   | 0.006245613098144531       |
| Torch   | False    | Train   | 0.9844444444444445 | 0.0762557789205755   | 0.041452392083528126 | 6.699035882949829          |
| Torch   | False    | Test    | 0.73               | 0.09256661991584858  | 0.05446470313230484  | 0.00738072395324707        |
| Torch   | True     | Train   | 0.9977777777777778 | 0.07428185548802657  | 0.03875123370214533  | 11.321129083633423         |
| Torch   | True     | Test    | 0.72               | 0.09256661991584858  | 0.07059373539036935  | 0.006941795349121094       |

**German dataset with *foreignWorker* sensitive feature** 

| Backend | Debiased | Dataset | Accuracy           | DEO                  | DAO                  | Train/Inference Time (sec) |
|---------| -------- | ------- |--------------------|----------------------|----------------------|----------------------------|
| TF2     | False    | Train   | 0.7877777777777778 | 0.1457682380111397   | 0.11333378617035725  | 11.467361211776733         |
| TF2     | False    | Test    | 0.76               | 0.38144329896907214  | 0.27835051546391754  | 0.007477998733520508       |
| TF2     | True     | Train   | 0.8233333333333334 | 0.10827333242765935  | 0.11014128515147402  | 35.2807183265686           |
| TF2     | True     | Test    | 0.75               | 0.3917525773195877   | 0.28350515463917525  | 0.007942438125610352       |
| Torch   | False    | Train   | 0.9877777777777778 | 0.1929764977584567   | 0.10715256079337046  | 7.042952299118042          |
| Torch   | False    | Test    | 0.72               | 0.4226804123711341   | 0.2989690721649485   | 0.007213592529296875       |
| Torch   | True     | Train   | 0.9955555555555555 | 0.18951229452520035  | 0.09706561608477107  | 12.220382928848267         |
| Torch   | True     | Test    | 0.69               | 0.4226804123711341   | 0.31443298969072164  | 0.006900787353515625       |
