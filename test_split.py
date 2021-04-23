import pandas as pd
import numpy as np

train = np.array(pd.read_csv('./train_data.csv',header =None))

batch_data = train[1]


# attr_batch = [x[2][1:-1].split() for x in batch_data]

attr_batch2 = batch_data[2][1: -1].split()

print()