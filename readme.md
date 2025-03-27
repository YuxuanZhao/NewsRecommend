# Troubleshoot CUDA

```shell
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

# NDCG@5

| model | NDCG@5 |
| ----- | ------ |
| better embedding | 0.8866 |
| category_id retrieval | 0.4506 |
| DIN | 0.2605 |
| XGBoost | 0.1176 |

| centroid | ground truth | min | max |
| -------- | ------------ | --- | --- |
| 325 | 14265 | 433 | 4672 |
| 300 | 15564 | 400 | 4974 |
| 275 | 14752 | 142 | 4240 |
| 250 | 14275 | 161 | 4385 |