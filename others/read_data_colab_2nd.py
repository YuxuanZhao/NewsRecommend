def data_offline(df_train_click, df_test_click):

    for col in df_train_click.columns:
        df_train_click[col] = df_train_click[col].astype(int)

    for col in df_test_click.columns:
        df_test_click[col] = df_test_click[col].astype(int)

    train_users = df_train_click['user_id'].unique().tolist()
    # 随机采样出一部分样本
    sample_size = min(50000, len(train_users))
    val_users = sample(train_users, sample_size)
    print(f'val_users num: {len(val_users)}')

    # 训练集用户 抽出行为数据最后一条作为线下验证集
    click_list = []
    valid_query_list = []
    groups = df_train_click.groupby(['user_id'])

    for user_id, g in tqdm(groups):
        if user_id in val_users:
          if len(g) > 1:
            valid_query = g.tail(1)
            valid_query_list.append(valid_query[['user_id', 'click_article_id']])
            train_click = g.head(g.shape[0] - 1)
            click_list.append(train_click)
          else:
            click_list.append(g)
        else:
            click_list.append(g)

    if not valid_query_list:
        print("Warning: valid_query_list is empty. No validation queries found.")
        # Create an empty DataFrame with the right columns to avoid the error
        df_valid_query = pd.DataFrame(columns=['user_id', 'click_article_id'])
    else:
        df_valid_query = pd.concat(valid_query_list, sort=False)

    df_train_click = pd.concat(click_list, sort=False)

    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list, columns=['user_id', 'click_article_id'])
    df_query = pd.concat([df_valid_query, df_test_query], sort=False).reset_index(drop=True)
    df_click = pd.concat([df_train_click, df_test_click], sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id','click_timestamp']).reset_index(drop=True)

    print(f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    print(f'{df_query.head()}')
    print(f'{df_click.head()}')

    # 保存文件
    os.makedirs('drive/MyDrive/user_data/data/offline', exist_ok=True)

    df_click.to_pickle('drive/MyDrive/user_data/data/offline/click.pkl')
    df_query.to_pickle('drive/MyDrive/user_data/data/offline/query.pkl')

df_train_click = pd.read_csv('drive/MyDrive/tcdata/train_click_log.csv')
df_test_click = pd.read_csv('drive/MyDrive/tcdata/testA_click_log.csv')
data_offline(df_train_click, df_test_click)

def csvInfo():
    df = pd.read_csv('movies_sm/ratings.csv')
    df.info()
    print('# userId:',  df.userId.nunique())
    print('# movieId:', df.movieId.nunique())
    print('rating distribution:', df.rating.value_counts())
    print('shape:', df.shape)