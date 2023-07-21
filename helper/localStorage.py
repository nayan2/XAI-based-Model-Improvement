import pickle
import os
__FILE_NAME__ = './data/report.pkl'

def read_pkl(file_name: str = __FILE_NAME__):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

def save_to_pkl(data: any, file_name: str = __FILE_NAME__):
    updated_data = {}
    if isinstance(data, dict):
        if os.path.exists(file_name):
            updated_data = read_pkl(file_name)
        updated_data.update(data)
    else:
        updated_data = data
    
    with open(file_name, 'wb') as handle:
        pickle.dump(updated_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def clean_pkl_file(file_name: str = __FILE_NAME__):
    with open(file_name, 'wb') as handle:
        pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
