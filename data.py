import pandas as pd
import numpy as np

data = pd.read_csv("./data/Microsoft_2015-2025.csv")
data = data.rename(columns={'Close/Last': 'Close'})

data['Close'] = pd.to_numeric(data['Close'].str.replace('$', '').str.replace(',', ''), errors='coerce')

data['log_returns'] = np.log(data['Close']).diff()
std = data['log_returns'].std() * np.sqrt(250)

print(std)
