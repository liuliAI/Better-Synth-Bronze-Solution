import json
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

# 自行更换模型路径
model = CLIPModel.from_pretrained("data/model/openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("data/model/openai/clip-vit-base-patch32")

# 使用正则表达式移除特殊标记
def clean_text(text):
    text = re.sub(r'<__dj__image> ', '', text)
    text = re.sub(r' <|__dj__eoc|>', '', text)
    return text.strip()

# 读取jsonl文件并计算相似度分数
def calculate_similarity_scores(jsonl_file):
    '''
    根据自身数据集结构进行不同的处理，以下代码适用于：
    {"id":"002369246","text":"<__dj__image> death at la fencede a detective's companion guide to the first crime mystery in this beloved series\n <|__dj__eoc|>","images":["\/data\/input\/pretrain_stage_1\/images\/00236\/002369246.jpg"]}
    {"id":"000798958","text":"<__dj__image> arthroplastic traction by robert boling, md arthroplastic traction therapy\n <|__dj__eoc|>","images":["\/data\/input\/pretrain_stage_1\/images\/00079\/000798958.jpg"]}
    {"id":"001525985","text":"<__dj__image> checklist of essential inventory management processes\n <|__dj__eoc|>","images":["\/data\/input\/pretrain_stage_1\/images\/00152\/001525985.jpg"]}
    '''
    similarity_scores = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            text = clean_text(data['text'])
            img_path = data['images'][0]  # 数据集中images是以列表的形式给出，如果有多条路径，只取第一个
            try:
                image = Image.open(img_path)
                inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity_scores.append(logits_per_image.item())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return similarity_scores

# 绘制直方图并计算
def plot_histogram_and_stats(scores):
# 根据数据集样本数量，设置不同的bins便于更好的展示
    plt.hist(scores, bins=30)
    plt.title('Histogram of Image-Text Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.show()

    max_score = np.max(scores)
    min_score = np.min(scores)
    avg_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"Max Score: {max_score}")
    print(f"Min Score: {min_score}")
    print(f"Average Score: {avg_score}")

    def k_sigma_interval(k):
        lower_bound = avg_score - k * std_score
        upper_bound = avg_score + k * std_score
        return (lower_bound, upper_bound)

    # 计算K sigma区间
    k = 1  # 可以修改为任意K值
    interval = k_sigma_interval(k)
    print(f"{k} sigma interval: {interval}")

def main(jsonl_file):
    similarity_scores = calculate_similarity_scores(jsonl_file)
    plot_histogram_and_stats(similarity_scores)

# 自行更换数据集路径
main("data/output/res.jsonl")

 