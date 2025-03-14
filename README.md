# NewsRecommend

Your task is to build a article recommendation model. You are given four csv files:

for test_click_log.csv, train_click_log.csv, they both have nine columns, and all field is integer: user_id, click_article_id, click_timestamp, click_environment, click_deviceGroup, click_os, click_country, click_region, click_referrer_type

for articles.csv, it has four columns, and all field is integer: article_id, category_id, created_at_ts, words_count

for articles_emb.csv, it has an integer field article_id, and 250 float embedding field named emb_0, emb_1, emd_2 ~ emb_249

You can only train on train_click_log.csv, articles.csv, articles_emb.csv. Noted the user_id in test_click_log will not appear in train_click_log and the user_id in train_click_log also will not appear in test_click_log. And you need to use test_click_log.csv to validate the model performance by Mean Reciprocal Rank, which means you need to use the lsat click_article_id of every user as the ground truth and predict five most preferable articles in order, comparing whether the ground truth appear in these five prediction and its rank to give a score to these prediction.

The model should contain three steps: process the data, candidate generation, ranking.





评价指标是 Mean Reciprocal Rank：实际发生的点击在预测的顺位中越靠前得分越高

recall (itemCF, userCF) -> ranking

余弦相似度

You are given four csv files: