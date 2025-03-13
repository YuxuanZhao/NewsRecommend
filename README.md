# NewsRecommend

testA_click_log.csv, train_click_log.csv
- [ ] user_id
- [ ] click_article_id
- [ ] click_timestamp
- [ ] click_environment
- [ ] click_deviceGroup
- [ ] click_os
- [ ] click_country
- [ ] click_region
- [ ] click_referrer_type

articles.csv
- [ ] article_id
- [ ] category_id
- [ ] created_at_ts
- [ ] words_count

articles_emb.csv
- [ ] article_id
- [ ] emb_0 ~ emb_249

评价指标是 Mean Reciprocal Rank

实际发生的点击在预测的顺位中越靠前得分越高