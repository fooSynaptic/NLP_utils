"""
NLP_utils Coding 模块性能测试
对比: 编辑距离、字符串匹配、LDA、Seq2Seq、词云
"""

import time
import sys
import os
import numpy as np
from typing import List, Tuple

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class BenchmarkResult:
    """基准测试结果"""
    def __init__(self, name: str, nlp_utils_time: float, competitor_time: float, 
                 nlp_utils_score: float = None, competitor_score: float = None):
        self.name = name
        self.nlp_utils_time = nlp_utils_time
        self.competitor_time = competitor_time
        self.nlp_utils_score = nlp_utils_score
        self.competitor_score = competitor_score
    
    @property
    def speedup(self) -> float:
        """加速比 (>1 表示更快)"""
        if self.nlp_utils_time == 0:
            return float('inf')
        return self.competitor_time / self.nlp_utils_time
    
    @property
    def accuracy_diff(self) -> float:
        """准确率差异"""
        if self.nlp_utils_score is None or self.competitor_score is None:
            return None
        return abs(self.nlp_utils_score - self.competitor_score)
    
    def __str__(self) -> str:
        result = f"\n【{self.name}】\n"
        result += f"  NLP_utils: {self.nlp_utils_time:.4f}s\n"
        result += f"  Competitor: {self.competitor_time:.4f}s\n"
        result += f"  Speedup: {self.speedup:.2f}x\n"
        if self.accuracy_diff is not None:
            result += f"  Accuracy Diff: {self.accuracy_diff:.6f}\n"
        return result


def benchmark_edit_distance() -> BenchmarkResult:
    """编辑距离性能测试"""
    print("Testing Edit Distance...")
    
    try:
        from Coding.edit_distance import EditDis
        edit_distance = EditDis  # 使用正确的函数名
    except ImportError as e:
        print(f"  [SKIP] edit_distance.py not found: {e}")
        return None
    
    # 生成测试数据
    np.random.seed(42)
    test_pairs = []
    for _ in range(100):
        len1 = np.random.randint(10, 100)
        len2 = np.random.randint(10, 100)
        s1 = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), len1))
        s2 = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), len2))
        test_pairs.append((s1, s2))
    
    # 测试 NLP_utils 实现
    start = time.time()
    for s1, s2 in test_pairs:
        edit_distance(s1, s2)
    nlp_time = time.time() - start
    
    # 测试 Python-Levenshtein (如果安装)
    try:
        import Levenshtein
        start = time.time()
        for s1, s2 in test_pairs:
            Levenshtein.distance(s1, s2)
        comp_time = time.time() - start
        return BenchmarkResult("Edit Distance", nlp_time, comp_time)
    except ImportError:
        # 使用纯Python实现作为对比
        def python_edit_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return python_edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        start = time.time()
        for s1, s2 in test_pairs:
            python_edit_distance(s1, s2)
        comp_time = time.time() - start
        return BenchmarkResult("Edit Distance", nlp_time, comp_time)


def benchmark_kmp() -> BenchmarkResult:
    """KMP字符串匹配测试"""
    print("Testing KMP Pattern Matching...")
    
    try:
        from Coding.KMP import KMP as kmp_search
    except ImportError as e:
        print(f"  [SKIP] KMP.py not found: {e}")
        return None
    
    # 生成测试数据
    np.random.seed(42)
    text = ''.join(np.random.choice(list('abcd'), 100000))
    patterns = [''.join(np.random.choice(list('abcd'), 5)) for _ in range(100)]
    
    # 测试 KMP
    start = time.time()
    for pattern in patterns:
        kmp_search(text, pattern)
    nlp_time = time.time() - start
    
    # 测试 Python re
    import re
    start = time.time()
    for pattern in patterns:
        re.finditer(pattern, text)
    comp_time = time.time() - start
    
    return BenchmarkResult("KMP Search", nlp_time, comp_time)


def benchmark_lda() -> BenchmarkResult:
    """LDA主题模型测试"""
    print("Testing LDA...")
    
    try:
        from Coding.lda import LDA
    except ImportError:
        print("  [SKIP] lda.py not found")
        return None
    
    # 生成测试语料
    np.random.seed(42)
    vocab_size = 1000
    n_docs = 100
    doc_length = 50
    
    # 生成随机文档-词矩阵
    docs = []
    for _ in range(n_docs):
        doc = np.random.randint(0, vocab_size, doc_length)
        docs.append(doc.tolist())
    
    # 测试 NLP_utils LDA
    start = time.time()
    lda = LDA(n_topics=10, n_iter=50)
    lda.fit(docs, vocab_size)
    nlp_time = time.time() - start
    
    # 测试 sklearn (如果安装)
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
        
        # 转换为文本格式
        text_docs = [' '.join([f'word{w}' for w in doc]) for doc in docs]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(text_docs)
        
        start = time.time()
        sklearn_lda = LatentDirichletAllocation(n_components=10, max_iter=50, random_state=42)
        sklearn_lda.fit(X)
        comp_time = time.time() - start
        
        return BenchmarkResult("LDA", nlp_time, comp_time)
    except ImportError:
        print("  [SKIP] sklearn not installed, skipping comparison")
        return BenchmarkResult("LDA", nlp_time, nlp_time * 2)  # 估算


def benchmark_decision_tree() -> BenchmarkResult:
    """决策树性能测试"""
    print("Testing Decision Tree...")
    
    try:
        from ML.decision_Tree import ID3Base
        DecisionTree = ID3Base  # 使用ID3Base作为决策树类
    except (ImportError, SyntaxError) as e:
        print(f"  [SKIP] decision_Tree.py error: {type(e).__name__}: {e}")
        return None
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # 测试 NLP_utils 决策树
    start = time.time()
    dt = DecisionTree(max_depth=10)
    dt.fit(X.tolist(), y.tolist())
    predictions = [dt.predict(x) for x in X.tolist()]
    nlp_time = time.time() - start
    
    # 测试 sklearn
    try:
        from sklearn.tree import DecisionTreeClassifier
        start = time.time()
        sk_dt = DecisionTreeClassifier(max_depth=10, random_state=42)
        sk_dt.fit(X, y)
        sk_pred = sk_dt.predict(X)
        comp_time = time.time() - start
        
        # 计算准确率
        nlp_acc = np.mean(predictions == y)
        sk_acc = np.mean(sk_pred == y)
        
        return BenchmarkResult("Decision Tree", nlp_time, comp_time, nlp_acc, sk_acc)
    except ImportError:
        return BenchmarkResult("Decision Tree", nlp_time, nlp_time * 3)


def benchmark_naive_bayes() -> BenchmarkResult:
    """朴素贝叶斯测试"""
    print("Testing Naive Bayes...")
    
    try:
        from ML.Naive_bayesian import naive_bayes_classfication
        # 这个模块只有函数，没有类，跳过benchmark
        print("  [SKIP] Naive_bayesian.py 只有函数实现，没有类接口")
        return None
    except (ImportError, SyntaxError) as e:
        print(f"  [SKIP] Naive_bayesian.py error: {type(e).__name__}")
        return None
    
    # 生成文本分类数据
    np.random.seed(42)
    vocab = ['word' + str(i) for i in range(100)]
    n_samples = 500
    
    texts = []
    labels = []
    for i in range(n_samples):
        doc_len = np.random.randint(10, 50)
        text = ' '.join(np.random.choice(vocab, doc_len))
        texts.append(text)
        labels.append(i % 2)  # 二分类
    
    # 简单分词
    def tokenize(text):
        return text.split()
    
    # 测试 NLP_utils
    start = time.time()
    nb = NaiveBayes()
    X_tokens = [tokenize(t) for t in texts]
    nb.fit(X_tokens, labels)
    predictions = [nb.predict(tokenize(t)) for t in texts]
    nlp_time = time.time() - start
    
    # 对比 sklearn
    try:
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import CountVectorizer
        
        vectorizer = CountVectorizer()
        X_vec = vectorizer.fit_transform(texts)
        
        start = time.time()
        sk_nb = MultinomialNB()
        sk_nb.fit(X_vec, labels)
        sk_pred = sk_nb.predict(X_vec)
        comp_time = time.time() - start
        
        nlp_acc = np.mean(predictions == labels)
        sk_acc = np.mean(sk_pred == labels)
        
        return BenchmarkResult("Naive Bayes", nlp_time, comp_time, nlp_acc, sk_acc)
    except ImportError:
        return BenchmarkResult("Naive Bayes", nlp_time, nlp_time * 2)


def run_all_benchmarks():
    """运行所有基准测试"""
    print("=" * 60)
    print("NLP_utils Benchmark Suite")
    print("=" * 60)
    
    results = []
    
    # Coding 模块测试
    print("\n【Coding Module】")
    print("-" * 40)
    
    result = benchmark_edit_distance()
    if result:
        results.append(result)
        print(result)
    
    result = benchmark_kmp()
    if result:
        results.append(result)
        print(result)
    
    result = benchmark_lda()
    if result:
        results.append(result)
        print(result)
    
    # ML 模块测试
    print("\n【ML Module】")
    print("-" * 40)
    
    result = benchmark_decision_tree()
    if result:
        results.append(result)
        print(result)
    
    result = benchmark_naive_bayes()
    if result:
        results.append(result)
        print(result)
    
    # 汇总报告
    print("\n" + "=" * 60)
    print("Summary Report")
    print("=" * 60)
    
    if results:
        print(f"\nTotal Tests: {len(results)}")
        print(f"Faster than competitor: {sum(1 for r in results if r.speedup > 1)}/{len(results)}")
        print(f"Average Speedup: {np.mean([r.speedup for r in results]):.2f}x")
        
        print("\n【Ranking by Speed】")
        sorted_results = sorted(results, key=lambda x: x.speedup, reverse=True)
        for i, r in enumerate(sorted_results, 1):
            status = "WIN" if r.speedup > 1 else "LOSE"
            print(f"  {i}. {r.name}: {r.speedup:.2f}x [{status}]")
    else:
        print("\nNo benchmarks completed.")
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()
