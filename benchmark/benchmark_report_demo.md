# NLP_utils Benchmark Report (DEMO)

Generated: 2026-03-08 16:15:00

## 测试环境
- Python: 3.11
- CPU: AMD Ryzen 9
- Memory: 32GB

------------------------------------------------------------

## Coding Module 测试结果

| Test | NLP_utils Time | Competitor Time | Speedup | Status |
|------|----------------|-----------------|---------|--------|
| Edit Distance | 0.0523s | 0.0891s | 1.70x | WIN |
| KMP Search | 0.0012s | 0.0018s | 1.50x | WIN |
| LDA | 2.3412s | 1.8923s | 0.81x | LOSE |
| Decision Tree | 0.1234s | 0.0987s | 0.80x | LOSE |
| Naive Bayes | 0.0456s | 0.0321s | 0.70x | LOSE |

**Average Speedup: 1.10x**

------------------------------------------------------------

## ML Module 测试结果

| Test | Train Speedup | Predict Speedup | Accuracy Diff | Status |
|------|---------------|-----------------|---------------|--------|
| KD-Tree KNN | 0.85x | 1.20x | 0.00 | LOSE/WIN |
| Viterbi | 1.50x | 1.30x | 0.00 | WIN/WIN |

------------------------------------------------------------

## 详细分析

### 获胜项目
1. **Edit Distance** (1.70x)
   - 自定义实现比纯Python快 70%
   - 比 python-Levenshtein 慢 15%
   - 建议: 使用Cython加速

2. **KMP Search** (1.50x)
   - 比Python re模块快 50%
   - 适用于多次搜索场景

3. **Viterbi** (1.50x / 1.30x)
   - 训练和预测都比 hmmlearn 快
   - 适合大规模HMM解码

### 待优化项目
1. **LDA** (0.81x)
   - 比 sklearn 慢 20%
   - 原因: 纯Python实现 vs sklearn Cython
   - 建议: 使用Gibbs采样优化

2. **Decision Tree** (0.80x)
   - 特征选择效率较低
   - 建议: 使用NumPy向量化

3. **Naive Bayes** (0.70x)
   - 文本向量化较慢
   - 建议: 使用稀疏矩阵

------------------------------------------------------------

## 源码问题汇总

以下文件存在语法/运行时错误，需要修复：

| 文件 | 问题 | 建议 |
|------|------|------|
| ML/decision_Tree.py | 第205行 `<path>` 未定义 | 修改为实际文件路径 |
| ML/Naive_bayesian.py | 第70行调用未定义函数 | 函数名拼写错误，应为 `naive_bayes_classfication` |

------------------------------------------------------------

## 改进建议

1. **性能优化**
   - 使用 NumPy 向量化代替循环
   - 考虑 Cython/Numba 加速关键路径
   - 使用 sklearn 的底层实现参考

2. **代码质量**
   - 修复语法错误
   - 添加类型提示
   - 增加单元测试

3. **文档完善**
   - 每个算法添加复杂度分析
   - 提供使用示例
   - 添加 benchmark 徽章到 README

------------------------------------------------------------

## Summary

- Total Tests: 7
- Wins: 4 (57%)
- Win Rate: 57.1%
- Files with Errors: 2

**总体评价**: 小型算法实现有优势，复杂ML算法需优化
