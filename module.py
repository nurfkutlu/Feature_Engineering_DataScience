import pandas as pd
## Load Datasets
def load_dataset(data_url='C:/Users/Hp/Desktop/Data_Science/Feature_Engineering/datasets/titanic.csv'):
    data =pd.read_csv(data_url)
    return data