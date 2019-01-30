#
import os
os.sys.path.append('/Users/ajmd/code/Py_utils')

from smart_load import read_from_file

def dir_loader(dir_path):
	files = [os.path.join(dir_path, x) for x in os.listdir(dir_path)]

	return [read_from_file(file) for file in files]



