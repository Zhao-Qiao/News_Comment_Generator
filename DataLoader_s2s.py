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
import math
def changetime_list(time_lists):
    timedict={}
    time2id=time_lists
    time2id.sort()
    for i in range(len(time2id)):
        timedict[time2id[i]]=i+1
    time_fin=[timedict[idx] for idx in time_lists]
    return time_fin
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


class Examplegnn(object):  # 一个新闻--->若干句子&词--->构图，node-word&sentence，边word2sent，sent2word
    # a news-a graph
    # no entity
    """Class representing a train/val/test example for single-document extractive summarization."""
    def __init__(self, news, vocab, sent_max_len, label, time_list):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.
        #news 一个新闻文本，包含若干句子
        :param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: float, the popularity of this news
        """
        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # Store the original strings
        self.original_article_sents = news

        # Process the mews
        for sent in self.original_article_sents:
            article_words = sent
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w) for w in
                                        article_words])  # list of word ids; OOVs are represented by the id for UNK token

        self._pad_encoder_input(vocab.word2id('[PAD]'))  # pad操作
        # Store the label
        self.label = label  # 值
        self.time_list = time_list  # time

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return:
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


class Exampledataset(Dataset):
    def __init__(self, data_file,vocab,senti_vocab,flag,freq_path=None,hps=None,pkl_path=None):
        """
        :param data_root:   数据集路径
        """
        self.vocab = vocab
        self.senti_vocab = senti_vocab
        self.data_root = data_file
        self.sent_max_len=hps['sent_max_len']
        #----------read process-----------
        self.entity_list,self.news_list, self.time_list, \
        self.vnames_list ,\
        self.label_list,\
        self.label_score_list,\
        self.title_list,\
        self.title_score_list,self.news_sent_list = jsonloader(data_file,flag)
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
        self.time_list=changetime_list(self.time_list)
    
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


    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        G.add_nodes(w_nodes)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes)
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        G.ndata["dtype"] = torch.zeros(w_nodes)
        return wid2nid, nid2wid

    def catWordNode(self, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1
        return wid2nid, nid2wid

    def get_entity_dict(self, inputid, entity_list, vocab):
        entity_dict = []
        for sentid in inputid:
            for wid in sentid:
                if vocab.id2word(wid) in entity_list:
                    entity_dict.append(wid)
        return entity_dict

    def Create_contentGraph(self, input_pad, w2s_w , label,time_list, index):
        G = dgl.DGLGraph()  # content graph
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)
        # print(wid2nid,nid2wid)
        N = len(input_pad)
        word_list=[]
        word_list_len = 100
        # TODO: need to set the length of word_list properly
        for i in range(N):
            for j in range(len(input_pad[i])):
                if input_pad[i][j] not in word_list:
                    word_list.append(input_pad[i][j])
        if len(word_list)<word_list_len: #needs larger
            for i in range(word_list_len-len(word_list)):
                word_list.append(0)
        word_list = word_list[:word_list_len]
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        #creat doc node
        G.add_nodes(1)
        docid = w_nodes + N
        G.ndata["unit"][docid] = torch.ones(1) * 2
        G.ndata["dtype"][docid] = torch.ones(1) * 2
        G.set_e_initializer(dgl.init.zero_initializer)

        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            # print("i:",i ,"; w2s;", len(w2s_w) )
            sent_tfw = w2s_w[i]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9).astype(int) # box = 10, astype(int) may result in loss of accuracy
                    # print(tfidf_box)
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]

        #add edge  sentence-news
        for i in range(N):
            sent_nid = sentid2nid[i]
            G.add_edges(docid,sent_nid,
                        data={"tffrac": torch.LongTensor([0]), "dtype": torch.Tensor([1])})
            G.add_edges(sent_nid, docid,
                        data={"tffrac": torch.LongTensor([0]), "dtype": torch.Tensor([1])})

        G.nodes[docid].data["id"] = torch.LongTensor([index + 1])
        G.nodes[docid].data["label"] = torch.FloatTensor([label])  # [1]
        G.nodes[docid].data["time"] = torch.LongTensor([time_list])
        G.nodes[docid].data["word"] = torch.LongTensor([word_list])
        if(docid==0):
            print("err")
        return G

    def get_graph(self, index):
        news = self.news_sent_list[index]
        label = self.title_score_list[index]
        time_list = self.time_list[index]
        example = Examplegnn(news, self.vocab, self.sent_max_len, label, time_list)
        return example


    def __len__(self):
        return len(self.title_list)
        #return len(self.news_list)

    def __getitem__(self, index):
        """
        should return a sample and a content graph of the news body
        """
        graph = self.get_graph(index)
        input_pad = graph.enc_sent_input_pad
        # print(len(input_pad))
        label = graph.label
        time_list = graph.time_list
        w2s_w = self.w2s_tfidf[index]  # 可以准确得到对应的tfidf
        G = self.Create_contentGraph(input_pad, w2s_w, label, time_list, index)
        #news = self.news_list[index]
        title= self.title_list[index]
        entity=self.entity_list[index]
        newlist = self.news_list[index]
        for_label=self.for_label_list[index]
        back_label=self.back_label_list[index]
        for_freq=[self.word2freq[word] for word in for_label]
        back_freq=[self.word2freq[word] for word in back_label]
        
        # print(entity)
        # print(title)
        # print(for_label)
        # print(back_label)
        new_token = []
        for word in newlist:
            if self.senti_vocab.word2id(word) != 1:
                new_token.append(self.senti_vocab.word2id(word))
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
                  'back_comment_freq':back_freq+[1]}

        return sample, G,index
    
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
    batched_data = []
    # samples = batch_dic[1]
    # graphs, index = map(list, zip(*samples))
    # graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    # sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    # batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    samples, graphs, index = map(list,zip(*batch_dic))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
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
    for idx in sorted_index:
        dic = batch_dic[idx][0]
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
    ''''''
    res={}
    
    # 内容标题id batch
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

    return [res, batched_graph]


