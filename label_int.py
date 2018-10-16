import pandas as pd
import sys

def label_int(data, col_list):
	data = pd.read_csv(data, index_col=0)
	print(data.columns)
	for col in col_list:
		for j, item in enumerate(data.loc[:,col]):
			print(item)
			data.loc[j,col] = list(data.loc[:,col].unique()).index(item)

	print(data.head())

if __name__ == '__main__':
	label_int(sys.argv[1],sys.argv[2])
