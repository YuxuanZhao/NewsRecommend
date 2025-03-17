- DDP: 每个 GPU 上训练一个相同的 Model
- Distributed Model Parallelism DMP：shard sparse part of model across device, 然后用 DDP replicate dense parts across device

# JaggedTensor

长度不一的sequence, batch x item

```python
# User interactions:
# - User 1 interacted with 2 items
# - User 2 interacted with 3 items
# - User 3 interacted with 1 item
lengths = [2, 3, 1]
values = torch.Tensor([101, 102, 201, 202, 203, 301])  # Item IDs interacted with
jt = JaggedTensor(lengths=lengths, values=values)
```

# KeyedJaggedTensor

batch = lengths / keys

这个是被用在 `EmbeddingBagCollection` 和 `EmbeddingCollection`

```python
keys = ["user_features", "item_features"]
# Lengths of interactions:
# - User features: 2 users, with 2 and 3 interactions respectively
# - Item features: 2 items, with 1 and 2 interactions respectively
lengths = [2, 3, 1, 2]
values = torch.Tensor([11, 12, 21, 22, 23, 101, 102, 201])
# Create a KeyedJaggedTensor
kjt = KeyedJaggedTensor(keys=keys, lengths=lengths, values=values)
# Access the features by key
print(kjt["user_features"])
# Outputs user features
print(kjt["item_features"])
```

# Planner

会把 `EmbeddingBagCollection` 和 `EmbeddingCollection` 转化成 `ShardedEmbeddingCollection` 和 `ShardedEmbeddingBagCollection`

- Table-wise (TW): as the name suggests, embedding table is kept as a whole piece and placed on one rank.
- Column-wise (CW): the table is split along the emb_dim dimension, for example, emb_dim=256 is split into 4 shards: [64, 64, 64, 64].
- Row-wise (RW): the table is split along the hash_size dimension, usually split evenly among all the ranks.
- Table-wise-row-wise (TWRW): table is placed on one host, split row-wise among the ranks on that host.
- Grid-shard (GS): a table is CW sharded and each CW shard is placed TWRW on a host.
- Data parallel (DP): each rank keeps a copy of the table.

使用 sharding 会增加 GPU 见的沟通成本：需要获取其他 GPU 的 embedding 和 gradient（`all2all` communication）

# DistributedModelParallel

1. 设置 process groups 和 device type
2. 默认使用 EmbeddingBagCollectionSharder
3. 默认生成 sharding plan
4. 创建 ShardedEmbeddingCollection 来代替原本的 EmbeddingCollection
5. 包装 DistributedDataParallel 来达到数据也并行

# Optimizer

gradient sorting -> aggregation -> sparse optimizer

# Inference

- Quantization: low latency, small model size, few devices <- FBGEMM TBE
- C++ for latency <- compile to TorchScript
- sharding embedding