The training dataset (train_data.csv) consists of 4 columns separated by semicolon (;):
- ID
- Question
- Answer
- Category

The questions contain possible answers listed after the actual question. The start of this listing is indicated by a dash (-).

The pandas Python library can be used to load the csv file:
	import pandas as pd
	df = pd.read_csv("train_dataset.csv", header=None, encoding="iso-8859-1", sep=";")