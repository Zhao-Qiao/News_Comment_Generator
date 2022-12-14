import torch
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import json
import os
import sys
from utils.vocabulary import Vocab
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import dgl
import numpy as np
from collections import Counter
# from tools import jsonloader
from transformers import AutoTokenizer
import math

def jsonfreqread(freq_path,vocab):
    file = open(freq_path, 'r', encoding='utf-8')
    id2freq={}
    word2freq={}
    filereader = file.read()
    pop_dict = json.loads(filereader)
    for word in list(pop_dict.keys()):
        word2freq[word]=pop_dict[word]
        id2freq[vocab.word2id(word)]=pop_dict[word]
    return word2freq

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
    full_sent_list = []
    no_entity_cnt = 0
    no_news_cnt = 0
    for line in file.readlines():
        full_sent = ""
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
            for token in new:
                full_sent += token
        if(newlist == []):
            no_news_cnt += 1
            continue
        news_sent_list.append(news_sent)
        vnames = pop_dict['v_names']
        if flag=='train':
            label=pop_dict['label']
        else:
            label=pop_dict['label']
        if entity not in label:
            no_entity_cnt += 1
            continue
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
        full_sent_list.append(full_sent)
    print("No Entity Comment Counts:", no_entity_cnt)
    print("No News Counts:",no_news_cnt)
    return entity_list,news_list, date_list, vnames_list,label_list,label_score_list,title_list,title_score_list,news_sent_list,full_sent_list



class Exampledataset(Dataset):
    def __init__(self, data_file,vocab,senti_vocab,flag,freq_path=None,pkl_path=None):
        """
        :param data_root:   数据集路径
        """
        self.vocab = vocab
        self.senti_vocab = senti_vocab
        self.data_root = data_file
        #----------read process-----------
        self.entity_list,self.news_list, self.time_list, \
        self.vnames_list ,\
        self.label_list,\
        self.label_score_list,\
        self.title_list,\
        self.title_score_list,self.news_sent_list,self.full_sent_list = jsonloader(data_file,flag)
        if freq_path!=None:
            self.word2freq=jsonfreqread(freq_path,vocab)

        if flag=="valid":
            self.label_list=self.changenews2list(self.label_list)

        # self.tfidf_list = creattfidf(news_lists=self.news_list)
        # TODO: tfidf data is pre-computed and imported directly to dataset. should be more flexible
        with open(pkl_path,'rb') as f:
            self.w2s_tfidf = pickle.load(f)
        # self.w2s_tfidf = creattfidf(news_lists=self.news_list)

        #----------data preprocess--------
        # self.news_list = self.changenews2list(news_list)
        self.title_list = self.delword(self.title_list)
        self.for_label_list,self.back_label_list = self.entitylabelcreat(self.label_list,self.entity_list)
        self.tokenizer = AutoTokenizer.from_pretrained("techthiyanes/chinese_sentiment",cache_dir="./pretrained_model")
    def delword(self,label_list):
        cur_news_list=[]
        for label in label_list:
            comment=[]
            for word in label:
                if self.vocab.word2id(word)!=1:
                    comment.append(word)
            #因为加了开始符号和结束符号
            if len(comment)>497:
                comment=comment[0:497]
            else:
                comment=comment
            cur_news_list.append(comment)
        return cur_news_list

    def entitylabelcreat(self,labels,entitys):
        forward_context_list=[]
        backward_context_list=[]
        for i in range(len(labels)):
            label=labels[i]
            entity=entitys[i]
            if entity not in label:
                print('fuck')
            #entity必在label中出现，不然不构成上下文
            target_index=label.index(entity)
            # 根据entity位置划分前后文
            backward_context=label[0:target_index+1]
            # 逆序
            backward_context=backward_context[::-1]
            #forward_context=label[target_index:]
            forward_context_list.append(label)
            backward_context_list.append(backward_context)
        return forward_context_list,backward_context_list

    def changenews2list(self,news_list):
        cur_news_list=[]
        for news in news_list:
            tmp=[]
            for sent in news:
                tmp.extend(sent)
            news_token=[]
            for word in tmp:
                if self.vocab.word2id(word)!=1:
                    news_token.append(word)
            if len(news_token)>499:
                news_token=news_token[0:499]
            cur_news_list.append(news_token)
        return cur_news_list




    def __len__(self):
        return len(self.title_list)
        #return len(self.news_list)

    def __getitem__(self, index):
        """
        should return a sample and a content graph of the news body
        """
        #news = self.news_list[index]
        title= self.title_list[index]
        entity=self.entity_list[index]
        newlist = self.news_list[index]
        for_label=self.for_label_list[index]
        back_label=self.back_label_list[index]
        for_freq=[self.word2freq[word] for word in for_label]
        back_freq=[self.word2freq[word] for word in back_label]
        full_sent = self.full_sent_list[index]
        new_token = []
        for word in newlist:
            if self.senti_vocab.word2id(word) != 1:
                new_token.append(self.senti_vocab.word2id(word))
        tokenized_news = self.tokenizer(full_sent,padding=True)
        # 将entity作为首位的输入
        sample = {'title_token':[self.vocab.word2id(entity)]+[self.vocab.word2id(word) for word in title] ,
                  'new_token':new_token ,
                  'entity':[self.senti_vocab.word2id(entity)] ,
                  # 添加结束符
                  'for_comment_token':[self.vocab.word2id(word) for word in for_label]+[self.vocab.word2id('[STOP]')],
                  # 添加开始符
                  'back_comment_token':[self.vocab.word2id(word) for word in back_label]+[self.vocab.word2id('[START]')],
                  # 开始和STOP频率设置为1
                  'for_comment_freq':for_freq+[1],
                  'back_comment_freq':back_freq+[1],
                  'full_sent':full_sent}

        return sample
    
def starattentionmask(length):
    global_mask=\
        [torch.tensor([True]*(length))]+\
        [torch.tensor([True]) for _ in range(length-1)]
    local_mask=\
        [torch.tensor([True]*(2))]+\
        [torch.tensor([False]*(i)+[True]*(3)) for i in range(0,length-2)]+\
        [torch.tensor([False]*(length-2)+[True]*(2))]
    global_attention_mask=pad_sequence(global_mask,batch_first=True)
    local_attention_mask=pad_sequence(local_mask,batch_first=True)
    attention_mask=global_attention_mask+local_attention_mask
    return attention_mask

def collate_func(batch_dic):
    # 先sort graph， 再根据sorted index处理embedding， 再组合
    #[batchsize, 2]
    batch_len=len(batch_dic)
    src_ids_batch = []
    new_ids_batch = []
    entity_batch = []
    back_tgt_ids_batch = []
    for_tgt_ids_batch = []
    src_pad_mask_batch = []
    back_tgt_pad_mask_batch = []
    for_tgt_pad_mask_batch = []
    back_tgt_freq_batch = []
    for_tgt_freq_batch = []
    full_sent_list = []
    for idx in range(batch_len):
        dic = batch_dic[idx]
        entity_batch.append(torch.tensor(dic['entity']))
        src_ids_batch.append(torch.tensor(dic['title_token']))
        new_ids_batch.append(torch.tensor(dic['new_token']))
        back_tgt_ids_batch.append(torch.tensor(dic['back_comment_token']))
        for_tgt_ids_batch.append(torch.tensor(dic['for_comment_token']))
        src_pad_mask_batch.append(torch.tensor([True] * len(dic['title_token'])))
        back_tgt_pad_mask_batch.append(torch.tensor([True] * len(dic['back_comment_token'])))
        for_tgt_pad_mask_batch.append(torch.tensor([True] * len(dic['for_comment_token'])))
        back_tgt_freq_batch.append(torch.tensor(dic['back_comment_freq']))
        for_tgt_freq_batch.append(torch.tensor(dic['for_comment_freq']))
        full_sent_list.append(dic['full_sent'])
    ''''''
    tokenizer = AutoTokenizer.from_pretrained("techthiyanes/chinese_sentiment",cache_dir="./pretrained_model")
    tokenized_input_ids_batch = torch.tensor(tokenizer(full_sent_list,padding=True, truncation=True, max_length=128)['input_ids'])
    # 内容标题id batch
    res = {}
    res['src_ids'] = pad_sequence(src_ids_batch,batch_first=True)
    # 新闻列表id batch
    res['new_ids'] = pad_sequence(new_ids_batch,batch_first=True)
    res['new_ids'] = torch.LongTensor(res['new_ids'].numpy())
    
    # 新闻列表mask
    res['new_ids_mask'] = res['new_ids'] != 0
    
    # 实体集合
    res['entity'] = torch.LongTensor(torch.stack(entity_batch,dim = 0))
    
    # 后向目标序列
    res['back_tgt_ids']=pad_sequence(back_tgt_ids_batch,batch_first=True)
    
    # 前向目标序列
    res['for_tgt_ids']=pad_sequence(for_tgt_ids_batch,batch_first=True)

    # 新闻 padmask
    res['src_pad_mask']=~pad_sequence(src_pad_mask_batch,batch_first=True)

    # 后向 padmask
    res['back_tgt_pad_mask']=~pad_sequence(back_tgt_pad_mask_batch,batch_first=True)
    # 前向 padmask
    res['for_tgt_pad_mask']=~pad_sequence(for_tgt_pad_mask_batch,batch_first=True)

    # 后向下三角mask矩阵
    back_tgt_length=res['back_tgt_pad_mask'].shape[1]
    back_tgt_mask_batch=[torch.tensor([True]*(i+1)) for i in range(back_tgt_length)]
    res['back_tgt_mask']=~pad_sequence(back_tgt_mask_batch,batch_first=True)

    # 下三角mask矩阵
    for_tgt_length=res['for_tgt_pad_mask'].shape[1]
    for_tgt_mask_batch=[torch.tensor([True]*(i+1)) for i in range(for_tgt_length)]
    res['for_tgt_mask']=~pad_sequence(for_tgt_mask_batch,batch_first=True)

    # 稀疏注意力矩阵
    src_length=res['src_pad_mask'].shape[1]
    res['src_mask']=starattentionmask(src_length)

    # frq 等长pad
    res['back_tgt_freq']=pad_sequence(back_tgt_freq_batch,batch_first=True)
    res['for_tgt_freq']=pad_sequence(for_tgt_freq_batch,batch_first=True)

    # 位置索引 【1,2,3,4,pad,pad,pad...】
    for_tgt_pos_batch = [torch.LongTensor([i+1 for i in range(for_tgt_length)]) for _ in range(res['for_tgt_pad_mask'].shape[0])]
    res['for_tgt_pos'] = pad_sequence(for_tgt_pos_batch, batch_first=True)

    back_tgt_pos_batch = [torch.LongTensor([i+1 for i in range(back_tgt_length)]) for _ in range(res['back_tgt_pad_mask'].shape[0])]
    res['back_tgt_pos'] = pad_sequence(back_tgt_pos_batch, batch_first=True)

    return [res, tokenized_input_ids_batch]


