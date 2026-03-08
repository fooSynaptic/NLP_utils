# NLP_utils Benchmark Suite

性能测试与对比框架，用于验证 NLP_utils 各模块的效率和准确性。

## 测试模块

### 1. Coding 模块
- **edit_distance**: 编辑距离计算速度 vs Python-Levenshtein
- **KMP**: 字符串匹配 vs Python re 模块
- **lda**: LDA主题模型 vs sklearn
- **seq2seq**: 序列生成速度 vs PyTorch/TensorFlow
- **word_cloud**: 词云生成 vs wordcloud 库

### 2. ML 模块
- **decision_Tree**: 决策树训练/预测 vs sklearn
- **Naive_bayesian**: 朴素贝叶斯 vs sklearn
- **KD_tree**: K近邻搜索 vs sklearn KDTree
- **Viterbi**: HMM解码速度 vs hmmlearn

### 3. InfoRetrive 模块
- 检索指标计算 (Precision, Recall, F1, NDCG)

## 运行方式

```bash
# 运行所有测试
python benchmark/run_all.py

# 运行特定模块
python benchmark/coding_benchmark.py
python benchmark/ml_benchmark.py

# 生成报告
python benchmark/generate_report.py
```

## 结果解读

- **Speedup**: >1 表示比对比库快
- **Accuracy**: 与参考实现的差异
- **Memory**: 内存占用对比
