import json

def jsonloader(filename):
    # 将数据加载到一个列表中
    file = open(filename, 'r', encoding='utf-8')
    entity_list=[]#实体
    news_list = []#新闻
    date_list = []#日期
    vnames_list = []#新闻所含实体列表
    label_list=[]#评论
    title_list=[]#标题
    label_score_list=[]#评论情感分数
    for line in file.readlines():
        pop_dict = json.loads(line)
        entity=pop_dict['entity']
        date = pop_dict['date']
        news = pop_dict['news']
        vnames = pop_dict['v_names']
        label=pop_dict['label']
        title=pop_dict['title']
        #label_score=pop_dict['label_score']


        news_list.append(news)
        date_list.append(date)
        entity_list.append(entity)
        vnames_list.append(vnames)
        label_list.append(label)
        title_list.append(title)
        #label_score_list.append(label_score)
    return entity_list,news_list, date_list, vnames_list,label_list,label_score_list,title_list
