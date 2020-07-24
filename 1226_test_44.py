
'''
完整测试 chinanews11.csv  0.798
        chinanews44.csv  0.774
        chinanews33.csv  0.754
jieba分词和 去停用词在一个函数内
'''

import numpy as np
import pandas as pd
import os
import time
import jieba
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from jieba import analyse
import gensim #自然语言处理库
from gensim import corpora,models,similarities
from sklearn.feature_extraction.text import CountVectorizer #词集转换成向量
from sklearn.feature_extraction.text import TfidfVectorizer #另一个转换成向量的库
from sklearn.naive_bayes import MultinomialNB #朴素贝叶斯多分类
from sklearn.metrics import classification_report

def read_file():
    data = pd.read_csv('test.csv',names=['category','theme','URL','content'])
    df = data.groupby('category').count()#展示数据规模
    print(df)
    return data

def split_content(data):
    content_raw = data['content'].str[:400]
    dict_content = {'content':content_raw.values}
    content = pd.DataFrame(dict_content)
    content.to_csv('uu_split_result.csv',header=None, index=False, encoding="utf-8-sig")
    data_1 = pd.read_csv('uu_split_result.csv',names=['content'])
    return data_1

#使用jieba分词,并去掉停用词
def separate_words(data):
    content = data.content.values.tolist() #将文本内容转换为list格式
    stopwords = pd.read_csv("stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8') #list
    stopwords = stopwords.stopword.values.tolist()
    print("正在分词,请耐心等候......")
    contents_clean = []
    all_words = []
    count = 1
    for line in content:
        #记录停用词读取进度
        process = str(count/1004710)
        count = count + 1 
        print(str(count-1)+"----进度:"+process)

        current_segment = jieba.lcut(line) #jieba分词
        #current_segment = [x.strip() for x in current_segment if x.strip()!='']
        if len(current_segment) > 1 and current_segment != "\r\n":
            line_clean = []
            for word in current_segment:
                if word in stopwords:
                    continue
                line_clean.append(word)
                all_words.append(str(word))
            contents_clean.append(line_clean)        
    print('------------分词完成-----------')
    return contents_clean, all_words

def format_transform(x): #x是数据集（训练集或者测试集）
    words =[]
    for line_index in range(len(x)):
        try:
            words.append(" ".join(x[line_index]))
        except:
            print("数据格式有问题")
    return words

def vec_transform(words):
    vec = CountVectorizer(analyzer="word",max_features=4000,ngram_range=(1, 3),lowercase=False)
    return vec.fit(words)

def main():
    data = read_file()
    data_1= split_content(data)
    contents_clean,all_words = separate_words(data_1)
    del data_1 #内存报错,所以及时删除变量

    #将清洗完的数据结果转化成DataFrame格式
    df_content = pd.DataFrame({"contents_clean":contents_clean})
    df_all_words = pd.DataFrame({"all_words":all_words})
    print(df_content.head())
    print("-------------------------------------")
    print(df_all_words.head())

    #LDA建模
    # dictionary = corpora.Dictionary(contents_clean) ##格式要求：list of list形式，分词好的的整个语料
    # corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]  #语料
    # lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20) #类似Kmeans自己指定K值
    # print (lda.print_topic(1, topn=10))
    # for topic in lda.print_topics(num_topics=10, num_words=10):
    #     print (topic[1])

    #打印DataFreme格式的内容和标签
    df_train = pd.DataFrame({"contents_clean":contents_clean,"label":data["category"]})
    df_train = shuffle(df_train)
    print("--------------------------------------1------------------------------------------")
    print(df_train.label.unique()) #打印标签的种类
    print("--------------------------------------2------------------------------------------")
      
    #标签转换
    label_mappping = {'汽车':1,'财经':2, '法治':3, '社会':4, '体育':5, '国际':6, '文化':7, '军事':8, '娱乐':9, '台湾':0}
    df_train["label"] = df_train["label"].map(label_mappping)
    print(df_train.head())
    print("--------------------------------------3------------------------------------------")

    #切分数据集
    x_train,x_test,y_train,y_test = train_test_split(df_train["contents_clean"].values,df_train["label"].values,test_size=0.5)
    #训练
    start_1 = time.time()#记录开始时间
    words_train = format_transform(x_train) 
    vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,ngram_range=(1, 3),lowercase = False)
    vectorizer.fit(words_train)#转为向量格式
    classifier = MultinomialNB()
    classifier.fit(vectorizer.transform(words_train), y_train)
    end_1 = time.time()
    #测试
    start_2 = time.time()
    words_test = format_transform(x_test)
    score = classifier.score(vectorizer.transform(words_test), y_test)
    end_2 = time.time()

    print("----------------------------------分类结果报告-----------------------------------------")
    print("分类准确率:" + str(score))
    print("训练时间:" + str(round((end_1-start_1), 2)) + '秒')
    print("测试时间:" + str(round((end_2-start_2), 2)) + '秒')
    y_predict=classifier.predict(vectorizer.transform(words_test))
    print(classification_report(y_test,y_predict))

if __name__ == "__main__":
    main()



