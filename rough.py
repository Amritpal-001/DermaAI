
import pandas as pd

def append_ext(fn):
    return fn+".jpg"
csv_file = '/home/amritpal/PycharmProjects/100-days-of-code/100_days_of_code/Skin_lesions_Classification-master/isic_2020/train.csv'

traindf=pd.read_csv( csv_file ,dtype=str)
print(traindf.head())

traindf["image_name"] = traindf["image_name"].apply(append_ext)

print(traindf.head())

traindf.to_csv(csv_file ,index=False)
#testdf["id"]=testdf["id"].apply(append_ext)