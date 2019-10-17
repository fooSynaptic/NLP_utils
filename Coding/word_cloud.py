# -*- coding: utf-8 -*-
__author__ = 'leilu'
__reference__ = 'https://blog.csdn.net/vivian_ll/article/details/68067574'
#wordcloud生成中文词云

from wordcloud import WordCloud
import codecs
import jieba
#import jieba.analyse as analyse
from scipy.misc import imread
import os
from os import path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

'''
wordcloud = WordCloud(font_path="simsun.ttf").generate(mytext)
%pylab inline
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
'''

import os
import fool
import jieba
from collections import Counter


# 绘制词云
def draw_wordcloud(file):
    print(file)
    #读入一个txt文件
    text = ''.join([x.strip() for x in open(file).readlines()])
    ners = fool.analysis(text)[1][0]
    comment_text = [x[3] for x in ners if x[2] in ['org', 'company']]
    #comment_text = open(path,'r').read()
    #结巴分词，生成字符串，如果不通过分词，无法直接生成正确的中文词云
    #cut_text = " ".join(jieba.cut(comment_text))
    cut_text = ' '.join(comment_text)
    d = path.dirname(__file__) # 当前文件文件夹所在目录
    color_mask = imread("/Users/ajmd/Desktop/timg.jpeg") # 读取背景图片
    cloud = WordCloud(
        #设置字体，不指定就会出现乱码
        #font_path="/Users/ajmd/Desktop/1252935991/CUHEISJ.TTF",
        font_path="/Users/ajmd/Desktop/simsunttc/simsun.ttc",
        #font_path=path.join(d,'simsun.ttc'),
        #设置背景色
        background_color='white',
        #词云形状
        mask=color_mask,
        #允许最大词汇
        max_words=2000,
        #最大号字体
        max_font_size=40
    )
    word_cloud = cloud.generate(cut_text) # 产生词云
    print("Done")
    word_cloud.to_file("./ads_cloud4.jpg") #保存图片
    #  显示词云图片
    
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    root_path = '/Users/ajmd/Documents/channel_ads'
    files = [os.path.join(root_path, x) for x in os.listdir(root_path)]
    draw_wordcloud(files[0])
