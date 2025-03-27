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

Best is trial 19 with value: 0.8866231619694659.
Best: {'lr': 0.0016221188831567399, 'weight_decay': 8.964506615010346e-05, 'attn_units': 128, 'fc_units': 32, 'dropout_rate': 0.3626827881888008, 'batch_size': 64, 'max_history': 64}