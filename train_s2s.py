# -*- coding: utf-8 -*-
import os
import time
import math
from utils.Word2Vec_emb import Word_Embedding
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from DataLoader_s2s import Exampledataset,collate_func
# from model import S2sTransformer,seq_generation_loss
# from transformerEncoderDecoder import S2sTransformer, seq_generation_loss
# from LSTM_EncoderDecoder_Albert import S2sLSTM, seq_generation_loss
from LSTM_EncoderDecoder import S2sLSTM, seq_generation_loss
from utils.vocabulary import Vocab
import numpy as np
import json
import matplotlib.pyplot as plt
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
def get_entity_list(entity_file_path):
    data = []
    with open(entity_file_path, 'r') as f:
        data = f.readlines()
        data = [i.rstrip() for i in data]
        f.close()
    return data
def creat_entitydict(entity_list,vocab):
    # FIXED: TODO : result error: entities are usually consists of more than 2 words, therefore they are often not included in the vocab
    entity_dict = {}
    entity_new_dict = {}
    index=0
    for entity in entity_list:
        entity_dict[entity]=index
        entity_new_dict[index] = vocab.word2id(entity)
        index+=1
    return entity_dict,entity_new_dict

def creat_entitymap(G,vocab,entity_dict,entseqlen):
    # TODO: fix bug: all entitiy maps are [0,0,0,0,0]
    Nnode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 2).tolist()
    entity_map = []
    newsid2nid={}
    for i in list(entity_dict.keys()):
        entity_map.append([])
    for idx in Nnode_id:
        word_list=G.nodes[idx].data["word"].squeeze().tolist()
        word_list = list(set(word_list))
        for word in word_list:
            if vocab.id2word(word) in list(entity_dict.keys()):
                news_id=int(G.nodes[idx].data["id"].item())
                newsid2nid[news_id]=idx
                entity_map[entity_dict[vocab.id2word(word)]].append(news_id)
    entity_map_pad=[]
    for i in range(len(entity_map)):
        seq_words = entity_map[i].copy()
        seqtimelist=[G.nodes[newsid2nid[ide]].data["time"].item() for ide in seq_words]
        data = list(zip(seq_words, seqtimelist))

        data.sort(key=lambda x : x[1])
        seq_words=[]
        for seq,times in data:
            seq_words.append(seq)
        if entity_map[i]!=[]:
            if len(seq_words) > entseqlen:

                seq_words = seq_words[:entseqlen]
            if len(seq_words) < entseqlen:
                seq_words.extend([0] * (entseqlen - len(seq_words)))
        else:
            seq_words.extend([0] * entseqlen)
        entity_map_pad.append(seq_words)
    return entity_map_pad

    '''
    (batch_len, )
    '''
    graphs, index= map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]

def Train(args):
    train_path = args['trainset_path']
    dev_path   = args['validset_path']
    vocab_path = args['vocab_path']
    senti_vocab_path = args['senti_vocab_path']
    vocab_len = args['vocab_len']
    senti_vocab_len = args['senti_vocab_len']
    entity_list_path=args['entity_list_path']
    resume = args['resume']
    checkpoint_path = args['checkpoint_path']
    history_path = args['history_path']
    log_path = args['log_path']
    model_name = args['model_save_name']
    model_resume_name = args['model_resume_name']
    batch_size = args['batch_size']
    end_epoch = args['end_epoch']
    lr = args['lr']
    loss_check_freq = args['loss_check']
    check_steps= args['check_steps']
    save_steps = args['save_steps']
    #os.environ['CUDA_VISIBLE_DEVICES'] = args['GPU_ids']
    embedding_path= args['embedding_path']
    word_emb_dim = args['word_emb_dim']
    nheads = args['nheads_transformer']
    minlr= args['minlr']
    lr_descent= args['lr_descent']
    pretrain_path= args['pretrain_path']
    senti_pretrain_path = args['senti_pretrain_path']
    GPU_ids=args['GPU_ids']
    numda=args['numda']
    gama=args['gama']
    freq_path=args['freq_path']
    sparse_attention=args['sparse_attention']
    devices = args['devices']
    ent_seq_len = args['ent_seq_len']
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ids

    #########
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(history_path):
        os.makedirs(history_path)
    log_save_name = 'log_' + model_name + '.log'
    logger = logging.getLogger(__name__)
    for k,v in args.items():
        logger.info(k+':'+ str(v))
    checkpoint_name = os.path.join(checkpoint_path, model_name + '_best_ckpt.pth')
    model_ckpt_name = os.path.join(checkpoint_path, model_name + '_best.pkl')

    if not model_resume_name:
        model_resume_name = model_ckpt_name
    localtime = time.asctime(time.localtime(time.time()))
    logger.info('#####start time:%s' % (localtime))
    time_stamp = int(time.time())
    logger.info('time stamp:%d' % (time_stamp))
    logger.info('######Model: %s' % (model_name))
    logger.info('trainset path ：%s' % (train_path))
    logger.info('valset path: %s' % (dev_path))
    logger.info('batch_size:%d' % (batch_size))
    logger.info('learning rate:%f' % (lr))
    logger.info('end epoch:%d' % (end_epoch))

    #加载生成词典w
    vocab_list=[]
    for line in open(vocab_path, "r"):
        vocab_list.append(line[:-1])
    print("[INFO] vocab_list读取成功！")
    print("[INFO] vocab_size:" , len(vocab_list))
    # 创建vocab类
    vocab = Vocab(vocab_list, vocab_len)
    # 创建entity_list
    entity_list = get_entity_list(entity_list_path)
    entity_dict, entity_news_dict = creat_entitydict(entity_list, vocab)
    #加载情感模型词典
    vocab_list2=[]
    for line in open(senti_vocab_path, "r"):
        vocab_list2.append(line[:-1])
    print("[INFO] senti_vocab_list读取成功！")
    print("[INFO] senti_vocab_size:" , len(vocab_list2))
    # 创建sentivocab类
    senti_vocab = Vocab(vocab_list2, senti_vocab_len)

    trainset = Exampledataset(train_path, vocab,senti_vocab, "train",freq_path,pkl_path=args['pkl_path'])

    print("训练集样本数：%d"%(trainset.__len__()))
    logger.info("训练集样本数：%d"%(trainset.__len__()))

    train_loader = DataLoader(trainset, batch_size=batch_size,num_workers=4, shuffle=True,collate_fn=collate_func,drop_last=True)

    # 部署GPU环境
    device = torch.device("cuda:"+devices if torch.cuda.is_available() else "cpu")
    gpu = False
    if torch.cuda.is_available(): gpu = True
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        print('Using GPU devices {}'.format(devices))

    # 加载 生成预训练embed
    if not os.path.exists(pretrain_path):
        print("无pretrain,加载中")
        embed_loader = Word_Embedding(embedding_path, vocab)
        vectors = embed_loader.load_my_vecs()
        pretrained_weight = embed_loader.add_unknown_words_by_uniform(vectors, word_emb_dim)
        np.save(pretrain_path,pretrained_weight)
        print("save完成")
    pretrained_weight = np.load(pretrain_path)
    pretrained_weight = torch.from_numpy(pretrained_weight).to(device)

    # 加载 情感预训练embed
    if not os.path.exists(senti_pretrain_path):
        print("无pretrain,加载中")
        embed_loader = Word_Embedding(embedding_path, senti_vocab)
        vectors = embed_loader.load_my_vecs()
        senti_pretrained_weight = embed_loader.add_unknown_words_by_uniform(vectors, word_emb_dim)
        np.save(senti_pretrain_path,senti_pretrained_weight)
        print("save完成")
    senti_pretrained_weight = np.load(senti_pretrain_path)
    # senti_pretrained_weight = torch.from_numpy(senti_pretrained_weight).to(device)
    senti_pretrained_weight = torch.from_numpy(senti_pretrained_weight)
    print("pretrain_weight load 成功！")
    ## some useful shits
    embed_for_gnn = torch.nn.Embedding(vocab.size(), word_emb_dim).to(device)
    ## end of shits
    # 模型上GPU，加载情感模型
    pretrained_weight = None
    # model = S2sLSTM(vocab=vocab , word_emb_dim=word_emb_dim, nhead=nheads,
    #                        pretrained_weight=pretrained_weight, device=device)
    model = S2sLSTM(vocab=vocab, senti_model = 'Albert', word_emb_dim = word_emb_dim, nhead = nheads, pretrained_weight = pretrained_weight, device=device)
    # model = S2sTransformer(vocab=vocab, senti_model=AFLSTM_model, word_emb_dim=word_emb_dim, nhead=nheads,768
    #                        pretrained_weight=pretrained_weight, entity_news_dict=entity_news_dict, hps=hps)
    model.to(device)
    device_ids = range(torch.cuda.device_count())

    # 从检查点加载模型，不使用
    if resume != 0:
        logger.info('Resuming from checkpoint...')
        model.load_state_dict(torch.load(model_resume_name))
        checkpoint = torch.load(checkpoint_name)
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
    else:
        best_loss =  math.inf
        start_epoch = -1
        history = {'train_loss': [], 'val_loss': []}

    # 加载loss、optim、scheduler
    criterion = seq_generation_loss(device=device,numda=numda,gama=gama).to(device)
    # criterion = seq_generation_loss(device=device, numda=numda, gama=gama)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = StepLR(optim, step_size=8, gamma=0.95)
    #每5个check对learning rate进行乘以0.95的衰减

    #training
    steps_cnt=0
    for epoch in range(start_epoch+1,end_epoch):
        print('-------------epoch:%d--------------'%(epoch))
        logger.info('-------------epoch:%d--------------'%(epoch))
        model.train()
        loss_tr = 0
        local_steps_cnt=0
        #########   train ###########
        print ('start training!')
        loss_list = []
        batch_list = []
        plt.figure()
        plt.ion()
        for batch_idx, batch_pair in tqdm(enumerate(train_loader),
                                     total=int(len(train_loader.dataset) / batch_size) + 1):
            batch = batch_pair[0]
            tokenized_ids = batch_pair[1]
            tokenized_ids = tokenized_ids.to(device)
            src_batch, \
            back_tgt_batch, \
            for_tgt_batch, \
            src_pad_mask, \
            back_tgt_pad_mask, \
            for_tgt_pad_mask, \
            back_tgt_att_mask, \
            for_tgt_att_mask, \
            src_mask_batch, \
            back_tgt_freq_batch, \
            for_tgt_freq_batch , \
            back_tgt_pos_batch, \
            for_tgt_pos_batch , \
            new_ids_batch , \
            new_ids_mask_batch , \
            entity_batch  =\
                batch['src_ids'], \
                batch['back_tgt_ids'], \
                batch['for_tgt_ids'], \
                batch['src_pad_mask'], \
                batch['back_tgt_pad_mask'], \
                batch['for_tgt_pad_mask'], \
                batch['back_tgt_mask'], \
                batch['for_tgt_mask'], \
                batch['src_mask'],\
                batch['back_tgt_freq'], \
                batch['for_tgt_freq'], \
                batch['back_tgt_pos'], \
                batch['for_tgt_pos'], \
                batch['new_ids'], \
                batch['new_ids_mask'], \
                batch['entity'], \

            src_batch = src_batch.to(device)
            back_tgt_batch = back_tgt_batch.to(device)
            for_tgt_batch = for_tgt_batch.to(device)
            src_pad_mask = src_pad_mask.to(device)
            back_tgt_pad_mask = back_tgt_pad_mask.to(device)
            for_tgt_pad_mask = for_tgt_pad_mask.to(device)
            back_tgt_att_mask = back_tgt_att_mask.to(device)
            for_tgt_att_mask = for_tgt_att_mask.to(device)
            src_mask_batch = src_mask_batch.to(device)
            back_tgt_freq_batch = back_tgt_freq_batch.to(device)
            for_tgt_freq_batch = for_tgt_freq_batch.to(device)
            back_tgt_pos_batch = back_tgt_pos_batch.to(device)
            for_tgt_pos_batch = for_tgt_pos_batch.to(device)
            new_ids_batch = new_ids_batch.to(device)
            new_ids_mask_batch = new_ids_mask_batch.to(device)
            entity_batch = entity_batch.to(device)
            model.zero_grad()

            # 不采用稀疏注意力矩阵计算
            if sparse_attention:
                for_out,back_out=model(src_ids = src_batch,
                                       back_tgt_ids = back_tgt_batch,
                                       for_tgt_ids = for_tgt_batch,
                                       src_pad_mask = src_pad_mask,
                                       back_tgt_pad_mask = back_tgt_pad_mask,
                                       for_tgt_pad_mask = for_tgt_pad_mask,
                                       back_tgt_mask = back_tgt_att_mask,
                                       for_tgt_mask = for_tgt_att_mask,
                                       src_mask = src_mask_batch,
                                       new_list = new_ids_batch,
                                       new_mask = new_ids_mask_batch,
                                       entity = entity_batch,
                                       memory_mask=None,
                                       gpunum = int(devices),
                                       tokenizer_id = tokenized_ids,

                                       )
            else:
                for_out,back_out=model(src_ids = src_batch,
                                       back_tgt_ids = back_tgt_batch,
                                       for_tgt_ids = for_tgt_batch,
                                       src_pad_mask = src_pad_mask,
                                       back_tgt_pad_mask = back_tgt_pad_mask,
                                       for_tgt_pad_mask = for_tgt_pad_mask,
                                       back_tgt_mask = back_tgt_att_mask,
                                       for_tgt_mask = for_tgt_att_mask,
                                       src_mask = None,
                                       new_list = new_ids_batch,
                                       new_mask = new_ids_mask_batch,
                                       entity = entity_batch,
                                       memory_mask=None,
                                       gpunum = int(devices),
                                       tokenizer_id=tokenized_ids)
            loss = criterion(for_out,back_out,
                             back_tgt_batch,
                             for_tgt_batch,
                             back_tgt_freq_batch,
                             for_tgt_freq_batch,
                             back_tgt_pos_batch,
                             for_tgt_pos_batch)
            #梯度回传
            loss.backward()  # compute gradients
            optim.step()  # update parameters
            #优化更新参数
            steps_cnt+=1
            local_steps_cnt+=1
            loss_tr += loss.item()

            # loss_list.append(loss.item())
            # batch_list.append(batch_idx)
            if batch_idx % loss_check_freq == 0:
                print('batch:%d' % (batch_idx))
                print('loss:%f' % (loss.item()))
                loss_list.append(loss.item())
                batch_list.append(batch_idx)
            if steps_cnt%check_steps == 0:
                loss_tr  /= local_steps_cnt
                print('trainset loss:%f' % (loss_tr))
                logger.info('trainset loss:%f' % (loss_tr))
                history['train_loss'].append(loss_tr)
                loss_tr = 0
                local_steps_cnt = 0

                scheduler.step()
                logger.info("current lr:%f" % (scheduler.get_last_lr()[0]))
                model.train()

            if steps_cnt%save_steps==0:
                logger.info('match save steps,Checkpoint Saving...')
                torch.save(model.state_dict(), './sport_ckpt/'+args['Model_Name']+'_step_'+str(steps_cnt)+'.pkl')
        loss_curv = plt.plot(batch_list, loss_list, 'r', lw=1)
        plt.title("loss")
        plt.xlabel('batch')
        plt.ylabel("loss")
        plt.show()
        #save model:
        logger.info('Save Model Every Epoch')
        print("Save Model on Epoch"+str(epoch))
        torch.save(model.state_dict(), './sport_ckpt/' +args['Model_Name']+'_Epochs_'+str(epoch)+'.pkl')
        if lr_descent:
            new_lr = max(minlr, lr / (epoch + 1))
            for param_group in list(optim.param_groups):
                param_group['lr'] = new_lr
            print("[INFO] The learning rate now is %f", new_lr)

if __name__ == "__main__":
    datasetname = "sport"
    args={
        "Model_Name":"LSTM",
        # 数据文件路径
        'trainset_path':"./data/generate_data/"+ datasetname + "_data/"+ datasetname + "_train_49500_2gram.json",
        'validset_path':"./data/generate_data/"+ datasetname + "_data/"+ datasetname + "_test_49500_2gram.json",
        # vocab路径
        'vocab_path':"./data/generate_data/"+ datasetname + "_data/"+ datasetname + "_vocab_49500_2gram.txt",
        # senti_vocab路径
        'senti_vocab_path':"./data/sentiment_data/"+ datasetname + "_data/"+ datasetname + "_sentiment_vocab.txt",
        # 词典最大长度
        'vocab_len':110000,
        'senti_vocab_len':150000,
        #path to entity_list(v_names actually...)
        'entity_list_path': "./data/generate_data/"+datasetname+"_data/"+datasetname+"_vnames_list.txt",
        # 模型保存路径
        'checkpoint_path':'./ckpt/',
        'history_path':'./history/',
        # 情感分析模型路径
        'senti_model_path':"./senti_ckpt/"+datasetname+"_AFLSTM.pt",
        # 日志路径
        'log_path':'./log/',
        # 预训练词embedding模型路径
        'embedding_path':'./Word2Vec/word_embedding', 
        'word_emb_dim':128, # default: 128
        'nheads_transformer':4, # embed_dim % nheads_transformer == 0
        # 是否使用之前的checkpoint ，0则为不使用
        'resume':0,
        'model_save_name':'transback_00_2gram_itf_sport',
        # only test
        'model_resume_name':'',
        # 训练batch参数
        'batch_size':12,
        'end_epoch':50,
        'check_steps':1000,
        # 模型保存间隔步数
        'save_steps':50000,
        'lr':1e-4,
        # 输出loss间隔步数
        'loss_check':300,
        # only test
        'version_info':'use pretrained embed , encode_layers=6 model.train() revise',
        'GPU_ids':'1',
        # 学习率递减
        'lr_descent':False,
        # 最小学习率
        'minlr':5e-5,
        # 加载预训练好的embedding数据路径
        'pretrain_path':"data/generate_data/"+ datasetname + "_data/pretrained_weight_"+ datasetname + "_vocab_49500_2gram.npy",
        # 加载预训练好的senti_embedding数据路径
        'senti_pretrain_path':"data/sentiment_data/"+ datasetname + "_data/pretrained_weight_"+ datasetname + "_sentiment_vocab.npy",
        # 词频字典路径
        'freq_path': "data/generate_data/"+ datasetname + "_data/"+ datasetname + "_49500_2-gram_labelFre.json",
        #pkl_path:
        'pkl_path': "./pkldata/sport_tfidf.pkl",
        # 超参
        'numda':0.4,
        'gama':0,
        'sparse_attention':False,
        'devices':'1',
        'ent_seq_len': 5, #maximum length of entity enquence
    }
    
    Train(args)
