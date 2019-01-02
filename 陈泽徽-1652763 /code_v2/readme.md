# Readme

## Environment
- make sure you are under a Python 3 environment.
- package dependance
	- pandas==0.23.4
	- numpy==1.14.5
	- scikit-learn==0.19.1
	- tensorflow==1.8.0
	- statsmodels==0.9.0
	- matplotlib==2.2.3
	- Keras==2.1.3

## Usage

1. make sure you've put train.csv in data/ directory
2. run `pip install -r requirements.txt`
3. run `python pre_data.py` to generate multi-feature processed data
4. run `python vae_script.py` to generate anomaly data using VAE
5. run `python script.py` to test our model on all kpi dataset, the record will generated automatically in file: `result.txt` and `output.csv`.

ps. You can also run `python dnn.py num(1,2,3...)` to test our model on a specific kpi dataset.

## Result Checking

All of our result will be recorded in output.csv and result.txt.

You can run this following script to compute the average f1 score for whole dataset, its final score should be around 0.80.

```Python
import pandas as pd

output = pd.read_csv('output.csv')
data = pd.read_csv('data/train.csv')
kpi_group = set(data['KPI ID'])

kpi_len = {}
data_len = data.shape[0]
for kpi_id in kpi_group:
    kpi_data = data[data['KPI ID'] == kpi_id]
    kpi_ratio = kpi_data.shape[0] / data_len
    kpi_len[kpi_id] = kpi_ratio

avg_f1 = 0
for row in output.iterrows():
    avg_f1 += row[1][2] * kpi_len[row[1][0]]

print(avg_f1)

# our random test value is 0.8039888185065309
```
