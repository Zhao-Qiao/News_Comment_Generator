# encoding: utf-8
'''
@Time: 10:39
@FileName: computeTFIDF.py
@Author: Zhao Qi-ao
'''
import math
import json
from collections import Counter
import pickle


def jsonloader(filename,flag):
    # newslist 每一项是将所有的句子合并成一个长单词序列，为了传入F4需要将每个句子分开。
    # 将数据加载到一个列表中
    file = open(filename, 'r', encoding='utf-8')
    entity_list=[]#实体
    news_list = []#包含实体的新闻新闻
    news_sent_list=[]# 未拼接的news，格式为：[[sent1,sent2,[w1,w2,...m]],[news2],...,[newsn]]
    date_list = []#日期
    vnames_list = []#新闻所含实体列表
    label_list=[]#评论
    title_list=[]#标题
    # no use
    title_score_list = []# sentiment score of news story
    label_score_list=[]#评论情感分数
    for line in file.readlines():
        pop_dict = json.loads(line)
        # skipping news without news story
        if 'entity' not in pop_dict and flag=='train':
            continue
        entity = pop_dict['entity']
        date = pop_dict['date']
        news = pop_dict['news']
        news_sent = []
        newlist = []
        for new in news:
            if entity in new:
                newlist.extend(new)
                news_sent.append(new)
        if(newlist == []):
            continue
        news_sent_list.append(news_sent)
        vnames = pop_dict['v_names']
        if flag=='train':
            label=pop_dict['label']
        else:
            label=pop_dict['label']
        title=pop_dict['title']
        label_score=pop_dict['label_score']
        title_score = pop_dict['title_score']

        news_list.append(newlist)
        date_list.append(date)
        entity_list.append(entity)
        vnames_list.append(vnames)
        label_list.append(label)
        title_list.append(title)
        label_score_list.append(label_score)
        title_score_list.append(title_score)
    return entity_list,news_list, date_list, vnames_list,label_list,label_score_list,title_list,title_score_list,news_sent_list




def tfcaculate(word, news):
    result = Counter(news)
    num = result[word]
    return num/len(news)


# 计算word在全局的idf
def idfcaculate(word, news_list):
    num = 0
    for s in news_list:
        if word in s:
            num += 1
    return math.log(len(news_list) / (num + 1))

    # 按照news id列表创建每一个单词的在当前new下的tfidf

def creattfidf(news_lists):
    tfidf_list=[]
    count = 0
    for news_list in news_lists:
        w2s_w = {}
        countj = 0
        for new in news_list:
            sent_tfw = {}
            for word in new:
                tfidf = tfcaculate(word, new) * idfcaculate(word, news_list)
                sent_tfw[word] = tfidf
            w2s_w[countj] = sent_tfw
            countj+=1
        tfidf_list.append(w2s_w)
        count += 1
        if count % 5 == 0:
            print("step:", count)
    return tfidf_list
datasetname = "sport"
flag = "train"
data_file = "./data/generate_data/"+ datasetname + "_data/"+ datasetname + "_train_49500_2gram.json"
dst_path = "./pkldata/"+datasetname+"_tfidf.pkl"
entity_list,news_list, time_list, \
vnames_list ,\
label_list,\
label_score_list,\
title_list,\
title_score_list,news_sent_list = jsonloader(data_file,flag)
w2s_tfidf = creattfidf(news_lists=news_sent_list)
with open(dst_path,'wb') as f:
    pickle.dump(w2s_tfidf,f)
    f.close()
print("Done.")
