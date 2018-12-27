#!usr/bin/env python
#-*- coding:utf-8 -*-


def read_from_file(directions):
	decode_set = ['utf-8', 'gb18030', 'ISO-8859-2', 'gbk', 'Error']	#encode set
	#loop in encode set
	for k in decode_set:
		try:
			file = open(directions, 'r', encoding = k)
			readfile = file.read()	#exception while open with wring encoding
			#print("open file %s with encoding %s" %(directions,k))#打印读取成功
			#readfile = readfile.encode(encoding="utf-8",errors="replace")#若是混合编码则将不可编码的字符替换为"?"。
			file.close()
			break	
		except:
			if k == 'Error':
				raise Exception("%s had noway to decode"%directions)
			continue
	return readfile



