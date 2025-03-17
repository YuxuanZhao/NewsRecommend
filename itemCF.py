import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import math
import pickle
from os.path import exists

data_path='./news/train_click_log.csv'
save_path = './similarity/itemcf_i2i_sim.pkl'
sim_item_topk = 10
recall_item_num = 10

def get_all_click_df():
    click_df = pd.read_csv(data_path)
    click_df = click_df.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    click_df = click_df.sort_values('click_timestamp')
    return click_df

def get_user_item_time(click_df):
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    
    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    
    return user_item_time_dict # {user1: [item1: time1, item2: time2..]...}

def itemcf_sim(user_item_time_dict):
    i2i_sim = defaultdict(dict)
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for item_i, i_click_time in item_time_list:
            item_cnt[item_i] += 1
            for item_j, j_click_time in item_time_list:
                if item_i == item_j: continue
                i2i_sim[item_i][item_j] = i2i_sim[item_i].get(item_j, 0) + 1 / math.log(len(item_time_list) + 1)
                
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j]) # 同一个人点了 i 和 j
    
    pickle.dump(i2i_sim_, open(save_path, 'wb'))
    
    return i2i_sim_

def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    user_hist_items = user_item_time_dict[user_id]
    
    item_rank = defaultdict(int)
    for user, (item_i, click_time) in enumerate(user_hist_items):
        count = 0
        for item_j, wij in sorted(i2i_sim[item_i].items(), key=lambda x: x[1], reverse=True):
            if item_j in user_hist_items: continue
            count += 1
            item_rank[item_j] += wij
            if count == sim_item_topk: break
    
    if len(item_rank) < recall_item_num: # 热门物品补全
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items(): continue
            item_rank[item] = - i - 100 # 放在最后
            if len(item_rank) == recall_item_num: break
    
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        
    return item_rank

def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click

def print_five_items(dictionary):
    items = list(dictionary.items())
    
    num_to_print = min(2, len(dictionary))

    for i in range(num_to_print):
        key, value = items[i]
        print(f"User: {key}, Value (Article, Rank): {value}")

if __name__ == '__main__':
    user_recall_items_dict = defaultdict(dict)
    click_df = get_all_click_df()
    user_item_time_dict = get_user_item_time(click_df)
    if exists(save_path): i2i_sim = pickle.load(open(save_path, 'rb'))
    else: i2i_sim = itemcf_sim(user_item_time_dict)

    item_topk_click = get_item_topk_click(click_df, k=50)

    for user in tqdm(click_df['user_id'].unique()):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click)
    
    print_five_items(user_recall_items_dict)