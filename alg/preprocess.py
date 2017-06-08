import pandas as pd

def csv_to_records(csv_path):
	"""
	Input:
		csv_path: path of csv file
	Output:
		records: [{...}] list of dictionary
	"""
	# csv -> data_frame
	df = pd.read_csv(csv_path, header = 0)
	# data_frame -> list of dic
	return df.to_dict(orient='records')
