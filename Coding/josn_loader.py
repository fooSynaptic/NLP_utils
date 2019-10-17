import json


#file_path = '/Users/ajmd/data/rawcorpus/corpus_data/141_20181022_1000_channel_0.txt'


def loadField(file, text_field, start_field, end_field):
	f = open(file, encoding = 'utf-8')
	setting = json.load(f)
	#print(setting)
	texts = [s[text_field] for s in setting]
	start_points = [s[start_field] for s in setting]
	end_points =[s[end_field] for s in setting]
	f.close()

	res_pair = dict(zip(texts, tuple(zip(start_points, end_points))))

	return res_pair

#print(loadField(file_path, 'text', 'beginTime', 'endTime'))


def interval(info_dict, themeset):
	assert isinstance(info_dict, dict)
	assert isinstance(themeset, list)

	resset = set()
	for i in themeset:
		for k in info_dict.keys():
			if i in k:
				resset.add(info_dict[k])

	return resset




#res = loadField(file_path, 'text', 'beginTime', 'endTime')
#print(interval(res, ['香园', '箱女', '快板书', '拓拓', '周检', '朱俭', '茅善玉', '露香女', '你是我久久等待的那个人']))

'''
try:
	for i in ['香园', '箱女', '快板书', '拓拓', '周检', '朱俭', '茅善玉', '露香女', '你是我久久等待的那个人']:
		for k in res.keys():
			if i in k:
				print(k, ':', res[k])
				resset.add(res[k])
except:
	pass

print("final point:", resset)
'''