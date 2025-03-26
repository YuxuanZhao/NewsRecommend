# Troubleshoot CUDA

```shell
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

# NDCG@5

| model | NDCG@5 |
| ----- | ------ |
| DIN | 0.2502 |
| XGBoost | 0.1176 |

Best: {'lr': 0.0008423242031631318, 'weight_decay': 4.9986867866974926e-05, 'attn_units': 96, 'fc_units': 96, 'dropout_rate': 0.130790251600353, 'batch_size': 64, 'max_history': 32}