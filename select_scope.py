'''
1 拼接各个数据表格
2 数据预处理,检验数据规模
3 选取每组数据的前n条
'''
import numpy as np 
import pandas as pd
import os
import jieba

#拼接
def con_data():
    # df1 = pd.read_csv('chinanews00.csv',names=['category','theme','URL','content'])
    # df2 = pd.read_csv('chinanews11.csv',names=['category','theme','URL','content'])
    # data = pd.concat([df1,df2],axis=0,ignore_index=True) #拼接表格
    data = pd.read_csv('chinanews.csv',names=['category','theme','URL','content'])
    df = data.groupby('category').count()#展示数据规模
    print(df)
    print(data.shape)
    return data


#数据预处理
def clean(data):
    data_1 = data.dropna(axis=0, how='any')
    data_2 = data_1.drop_duplicates(keep='first')
    data_3 = data_2.drop(index=(data_2.loc[(data_2['category']=='category')].index))
    data_31 = data_3.drop(index=(data_3.loc[(data_3['category']=='教育')].index))
    data_32 = data_31.drop(index=(data_31.loc[(data_31['category']=='港澳')].index))
    data_4 = data_32.drop(index=(data_32.loc[(data_32['content']==' ')].index))

    df = data_4.groupby('category').count()
    print(data.shape)
    print(data_4.shape)
    print(df)
    return data_4

#分组选行
def group(data,amount,file_path):
    df = data.groupby('category').head(amount)
    df.to_csv(file_path,mode='a',header=None, index=False, encoding="utf-8-sig")

#将处理完的数据存储为新的csv文件
def save_file(data):
    #root = ".//newsCollection//"
    #path = root + "chinanews00.csv"#此处许修改
    path = "chinanews11.csv"
    try:
        # if not os.path.exists(root):
        #         os.mkdir(root)
        #         print('mkdir success')
        data.to_csv(path, header=None, index=False, encoding="utf-8-sig")
    except IOError:
        print('sorry, write failed')
    else:
        print("---chinanews00.csv have been added---")

def main():
    data = con_data() #表格拼接 
    # print('-------------------------')
    # data_pro = data_preprocessing(data) #数据预处理
    file_path = 'svm_data.csv'
    group(data,1000,file_path) #另外存储文件
    #df_content,df_all_words = separate_words(data_pro) #jieba分词
    
    
if __name__ == "__main__":
    main()
