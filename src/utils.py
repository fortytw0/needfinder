import json
import os
from tqdm import tqdm

def iterate(func, num_iter, pass_counter=False, message='', start_iter=0) : 

    returned_object = []

    for i in tqdm(range(start_iter, num_iter)) : 

        if pass_counter : 
            returned_object.append(func(i))

        else : 
            returned_object.append(func())

    return returned_object


def read_json(json_path, mode='r') : 

    assert mode in ['r' , 'rb'] , 'Only accepted modes for reading JSON : "r" or "rb" '

    with open(json_path , mode) as f : 
        return json.load(f)

def create_dir_if_not_exist(dir) : 

    if not os.path.isdir(dir) : 
        os.mkdir(dir)
        return False

    return True

def _read_jsonl_with_multiple_fields(jsonl_lines, fields) : 

    data = []

    for line in jsonl_lines : 
        try : 
            d = json.load(line) 
            data.append({f:d[f] for f in fields})
        except :
            pass

    return data

def _read_jsonl_with_single_field(jsonl_lines, field) : 

    data = []

    for line in jsonl_lines : 
        try : 
            d = json.load(line) 
            data.append(d[field])
        except : 
            pass

    return data

def _read_jsonl_with_all_fields(jsonl_lines) : 

    data = []

    for line in jsonl_lines : 
        try : 
            d = json.load(line)
            data.append(d)
        except : 
            pass

    return data

def read_jsonl(jsonl_path, field=None, fields=None, max_lines=None, mode='r') : 

    assert (field is None) or (fields is None) , 'Cannot pass one field and multiple fields to read.'

    with open(jsonl_path, mode) as f : 

        if max_lines is None : 
            lines = f.readlines()
                
        else : 
            lines = iterate(f.readline, max_lines)


    if field is not None : return _read_jsonl_with_single_field(lines, field)
                
    elif fields is not None : return _read_jsonl_with_multiple_fields(lines, fields)
        
    else : return _read_jsonl_with_all_fields(lines)






