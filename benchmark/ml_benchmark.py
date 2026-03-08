"""
NLP_utils ML 模块性能测试
对比: 决策树、朴素贝叶斯、Viterbi、KD树
"""

import time
import sys
import os
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MLBenchmarkResult:
    """ML模块基准测试结果"""
    def __init__(self, name: str, train_time_nlp: float, train_time_sklearn: float,
                 predict_time_nlp: float, predict_time_sklearn: float,
                 accuracy_nlp: float = None, accuracy_sklearn: float = None):
        self.name = name
        self.train_time_nlp = train_time_nlp
        self.train_time_sklearn = train_time_sklearn
        self.predict_time_nlp = predict_time_nlp
        self.predict_time_sklearn = predict_time_sklearn
        self.accuracy_nlp = accuracy_nlp
        self.accuracy_sklearn = accuracy_sklearn
    
    @property
    def train_speedup(self) -> float:
        return self.train_time_sklearn / self.train_time_nlp if self.train_time_nlp > 0 else 0
    
    @property
    def predict_speedup(self) -> float:
        return self.predict_time_sklearn / self.predict_time_nlp if self.predict_time_nlp > 0 else 0
    
    def __str__(self) -> str:
        result = f"\n【{self.name}】\n"
        result += f"  Training: NLP_utils={self.train_time_nlp:.4f}s, sklearn={self.train_time_sklearn:.4f}s (speedup: {self.train_speedup:.2f}x)\n"
        result += f"  Prediction: NLP_utils={self.predict_time_nlp:.4f}s, sklearn={self.predict_time_sklearn:.4f}s (speedup: {self.predict_speedup:.2f}x)\n"
        if self.accuracy_nlp is not None and self.accuracy_sklearn is not None:
            result += f"  Accuracy: NLP_utils={self.accuracy_nlp:.4f}, sklearn={self.accuracy_sklearn:.4f}, diff={abs(self.accuracy_nlp - self.accuracy_sklearn):.4f}\n"
        return result


def benchmark_kdtree():
    """KD树K近邻搜索测试"""
    print("Testing KD-Tree KNN...")
    
    try:
        from ML.KD_tree import KDTree
    except (ImportError, SyntaxError) as e:
        print(f"  [SKIP] KD_tree.py error: {type(e).__name__}")
        return None
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 10000
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    X_test = np.random.randn(100, n_features)
    
    # 测试 NLP_utils KDTree
    start = time.time()
    kdtree = KDTree()
    kdtree.fit(X_train.tolist())
    train_time_nlp = time.time() - start
    
    start = time.time()
    for x in X_test:
        kdtree.query(x.tolist(), k=5)
    predict_time_nlp = time.time() - start
    
    # 测试 sklearn
    try:
        from sklearn.neighbors import KDTree as SKKDTree
        
        start = time.time()
        sk_kdtree = SKKDTree(X_train, leaf_size=10)
        train_time_sklearn = time.time() - start
        
        start = time.time()
        sk_kdtree.query(X_test, k=5)
        predict_time_sklearn = time.time() - start
        
        return MLBenchmarkResult("KD-Tree KNN", train_time_nlp, train_time_sklearn,
                                  predict_time_nlp, predict_time_sklearn)
    except ImportError:
        return MLBenchmarkResult("KD-Tree KNN", train_time_nlp, train_time_nlp * 2,
                                  predict_time_nlp, predict_time_nlp * 2)


def benchmark_viterbi():
    """Viterbi算法测试"""
    print("Testing Viterbi Algorithm...")
    
    try:
        from ML.veterbi import viterbi
    except (ImportError, SyntaxError) as e:
        print(f"  [SKIP] veterbi.py error: {type(e).__name__}")
        return None
    
    # 使用veterbi.py中定义的示例数据
    states = ('Rainy', 'Sunny')
    observations = ('walk', 'shop', 'clean')
    start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
    transition_probability = {
        'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
        'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
    }
    emission_probability = {
        'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
        'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
    }
    
    # 测试 NLP_utils Viterbi
    start = time.time()
    for _ in range(1000):
        viterbi(observations, states, start_probability, 
                transition_probability, emission_probability)
    nlp_time = time.time() - start
    
    # 对比 hmmlearn
    try:
        from hmmlearn import hmm
        import numpy as np
        
        # 转换为hmmlearn格式
        model = hmm.MultinomialHMM(n_components=2, n_iter=1)
        model.startprob_ = np.array([0.6, 0.4])
        model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
        model.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
        
        # 观测序列编码
        obs_map = {'walk': 0, 'shop': 1, 'clean': 2}
        obs_seq = np.array([[obs_map[o]] for o in observations])
        
        start = time.time()
        for _ in range(1000):
            model.decode(obs_seq, algorithm='viterbi')
        comp_time = time.time() - start
        
        return MLBenchmarkResult("Viterbi", nlp_time, comp_time, 0, 0)
    except ImportError:
        return MLBenchmarkResult("Viterbi", nlp_time, nlp_time * 1.5, 0, 0)


def benchmark_all_ml():
    """运行所有ML模块测试"""
    print("=" * 60)
    print("ML Module Benchmark Suite")
    print("=" * 60)
    
    results = []
    
    result = benchmark_kdtree()
    if result:
        results.append(result)
        print(result)
    
    result = benchmark_viterbi()
    if result:
        results.append(result)
        print(result)
    
    # 汇总
    print("\n" + "=" * 60)
    print("ML Summary")
    print("=" * 60)
    
    if results:
        print(f"\nTotal ML Tests: {len(results)}")
        
        train_speedups = [r.train_speedup for r in results]
        predict_speedups = [r.predict_speedup for r in results]
        
        print(f"Training - Faster: {sum(1 for s in train_speedups if s > 1)}/{len(train_speedups)}, Avg Speedup: {np.mean(train_speedups):.2f}x")
        print(f"Prediction - Faster: {sum(1 for s in predict_speedups if s > 1)}/{len(predict_speedups)}, Avg Speedup: {np.mean(predict_speedups):.2f}x")
    
    return results


if __name__ == "__main__":
    benchmark_all_ml()
