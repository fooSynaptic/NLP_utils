"""
运行所有 benchmark 并生成报告
"""

import sys
import os
import json
from datetime import datetime

# 导入测试模块
sys.path.insert(0, os.path.dirname(__file__))

try:
    from coding_benchmark import run_all_benchmarks as run_coding
    from ml_benchmark import benchmark_all_ml as run_ml
except ImportError as e:
    print(f"Error importing benchmark modules: {e}")
    sys.exit(1)


def generate_markdown_report(coding_results, ml_results):
    """生成 Markdown 格式报告"""
    report = []
    report.append("# NLP_utils Benchmark Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("-" * 60)
    
    # Coding 模块结果
    if coding_results:
        report.append("\n## Coding Module\n")
        report.append("| Test | NLP_utils Time | Competitor Time | Speedup | Status |")
        report.append("|------|----------------|-----------------|---------|--------|")
        
        for r in coding_results:
            status = "WIN" if r.speedup > 1 else "LOSE"
            report.append(f"| {r.name} | {r.nlp_utils_time:.4f}s | {r.competitor_time:.4f}s | {r.speedup:.2f}x | {status} |")
        
        avg_speedup = sum(r.speedup for r in coding_results) / len(coding_results)
        report.append(f"\n**Average Speedup: {avg_speedup:.2f}x**")
    
    # ML 模块结果
    if ml_results:
        report.append("\n## ML Module\n")
        report.append("| Test | Train Speedup | Predict Speedup | Status |")
        report.append("|------|---------------|-----------------|--------|")
        
        for r in ml_results:
            train_status = "WIN" if r.train_speedup > 1 else "LOSE"
            predict_status = "WIN" if r.predict_speedup > 1 else "LOSE"
            report.append(f"| {r.name} | {r.train_speedup:.2f}x | {r.predict_speedup:.2f}x | {train_status}/{predict_status} |")
    
    # 总体评价
    report.append("\n## Summary\n")
    all_results = coding_results + ml_results
    if all_results:
        wins = sum(1 for r in all_results if getattr(r, 'speedup', 0) > 1 or getattr(r, 'train_speedup', 0) > 1)
        total = len(all_results)
        report.append(f"- Total Tests: {total}")
        report.append(f"- Wins: {wins}")
        report.append(f"- Win Rate: {wins/total*100:.1f}%")
    
    return "\n".join(report)


def main():
    """主函数"""
    print("=" * 70)
    print("NLP_utils Complete Benchmark Suite")
    print("=" * 70)
    
    # 运行 Coding 测试
    print("\n\n")
    coding_results = run_coding()
    
    # 运行 ML 测试
    print("\n\n")
    ml_results = run_ml()
    
    # 生成报告
    print("\n\n")
    print("=" * 70)
    print("Generating Report...")
    print("=" * 70)
    
    report = generate_markdown_report(coding_results, ml_results)
    
    # 保存报告
    report_file = os.path.join(os.path.dirname(__file__), 'benchmark_report.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_file}")
    print("\n" + "=" * 70)
    print("REPORT PREVIEW")
    print("=" * 70)
    print(report)
    
    # 保存 JSON 结果
    results_file = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'coding': [
            {
                'name': r.name,
                'nlp_utils_time': r.nlp_utils_time,
                'competitor_time': r.competitor_time,
                'speedup': r.speedup
            } for r in (coding_results or [])
        ],
        'ml': [
            {
                'name': r.name,
                'train_speedup': r.train_speedup,
                'predict_speedup': r.predict_speedup
            } for r in (ml_results or [])
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nRaw results saved to: {results_file}")


if __name__ == "__main__":
    main()
