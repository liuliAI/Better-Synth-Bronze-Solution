import json
import matplotlib.pyplot as plt
import numpy as np
import re

# 用于计算文本中的单词数量
def count_words(text):
    return len(text.split())

def calculate_word_counts(jsonl_file):
    word_counts = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            text = data['text']
            # 清除文本中的特殊标记
            text = re.sub(r'<__dj__image> ', '', text)
            text = re.sub(r' <|__dj__eoc|>', '', text)
            text = text.strip()
            word_counts.append(count_words(text))
    return word_counts

# 绘制直方图并计算统计数据
def plot_histogram_and_stats(word_counts):
# 根据数据集样本数量，设置不同的bins便于更好的展示
    plt.hist(word_counts, bins=30)
    plt.title('Histogram of Word Count in Texts')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.show()

    max_count = np.max(word_counts)
    min_count = np.min(word_counts)
    avg_count = np.mean(word_counts)
    std_count = np.std(word_counts)

    print(f"Max Word Count: {max_count}")
    print(f"Min Word Count: {min_count}")
    print(f"Average Word Count: {avg_count}")

    def k_sigma_interval(k):
        lower_bound = avg_count - k * std_count
        upper_bound = avg_count + k * std_count
        return (lower_bound, upper_bound)

    # 计算K sigma区间
    k = 1  # 可以修改为任意K值
    interval = k_sigma_interval(k)
    print(f"{k} sigma interval: {interval}")

def main(jsonl_file):
    word_counts = calculate_word_counts(jsonl_file)
    plot_histogram_and_stats(word_counts)

# 自行更换数据集路径
main("res.jsonl")
