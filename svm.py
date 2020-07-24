
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import jieba
import re

from collections import Counter
from wordcloud import WordCloud
#添加中文字体
from pylab import *
plt.rcParams['font.sans-serif'] = ['SimHei']

def read_file():
    data = pd.read_csv('svm_data.csv',names=['category','theme','URL','content'])
    df = data.groupby('category').count()#展示数据规模
    print(df)
    data = data[['category','content']]
    print(data)
    d={'category':data['category'].value_counts().index,'count':data['category'].value_counts()}
    df_cat=pd.DataFrame(data=d).reset_index(drop=True)
    print(df_cat)
    #图形化方式查看各类别分布
    df_cat.plot(x='category',y='count',kind='bar',legend=False,figsize=(8,5))
    plt.title('各类别数目分布图')
    plt.ylabel('数量',fontsize=18)
    plt.xlabel('类别',fontsize=18)
    plt.show()
    return data

def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line
 
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  


 
def generate_wordcloud(tup):
    wordcloud = WordCloud(background_color='white',
                          font_path='simhei.ttf',
                          max_words=50, max_font_size=40,
                          random_state=42
                         ).generate(str(tup))
    return wordcloud
 
def main():
    data=read_file()
    # #标签转换(方法一)
    # label_mappping = {'汽车':1,'财经':2, '法治':3, '社会':4, '体育':5, '国际':6, '文化':7, '军事':8, '娱乐':9, '台湾':0}
    # data['category'] = data['category'].map(label_mappping)
    # print(data.sample(10))
    #标签转换(方法二)
    data['cat_id']=data['category'].factorize()[0]
    cat_id_df = data[['category','cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
    cat_to_id = dict(cat_id_df.values)
    id_to_cat = dict(cat_id_df[['cat_id','category']].values)
    print(cat_id_df)
    #去停用词
    stopwords = stopwordslist("stopwords.txt")
    data['clean_content']=data['content'].apply(remove_punctuation)
    data['cut_content']=data['clean_content'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
    print(data.sample(10))
    #生成词云(未下载字体,无法显示)
    cat_desc = dict()
    for cat in cat_id_df.category.values: 
        text = data.loc[data['category']==cat, 'cut_content']
        text = (' '.join(map(str,text))).split(' ')
        cat_desc[cat]=text
        
    fig,axes = plt.subplots(5, 2, figsize=(30, 38))
    k=0
    for i in range(5):
        for j in range(2):
            cat = id_to_cat[k]
            most100=Counter(cat_desc[cat]).most_common(100)
            ax = axes[i, j]
            ax.imshow(generate_wordcloud(most100), interpolation="bilinear")
            ax.axis('off')
            ax.set_title("{} Top 100".format(cat), fontsize=30)
            k+=1
    ax.show()


if __name__ == "__main__":
    main()