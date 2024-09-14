import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 函数用于获取图片的宽度和高度
def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None

def calculate_image_dimensions(jsonl_file):
    widths = []
    heights = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            image_path = data['images'][0]
            width, height = get_image_dimensions(image_path)
            if width is not None and height is not None:
                widths.append(width)
                heights.append(height)
    return widths, heights

# 绘制直方图并计算统计数据
def plot_histogram_and_stats(dimension_data, dimension_name):
    plt.figure(figsize=(10, 5))
    # 根据数据集样本数量，设置不同的bins便于更好的展示
    plt.hist(dimension_data, bins=30)
    plt.title(f'Histogram of {dimension_name} Pixel Size')
    plt.xlabel(f'{dimension_name} Pixel Size')
    plt.ylabel('Frequency')
    plt.show()

    max_dim = np.max(dimension_data)
    min_dim = np.min(dimension_data)
    avg_dim = np.mean(dimension_data)
    std_dim = np.std(dimension_data)

    print(f"Max {dimension_name}: {max_dim}")
    print(f"Min {dimension_name}: {min_dim}")
    print(f"Average {dimension_name}: {avg_dim}")

    def k_sigma_interval(k):
        lower_bound = avg_dim - k * std_dim
        upper_bound = avg_dim + k * std_dim
        return (lower_bound, upper_bound)

    # 计算K sigma区间
    k = 1  # 可以修改为任意K值
    interval = k_sigma_interval(k)
    print(f"{k} sigma interval: {interval}")

def main(jsonl_file):
    widths, heights = calculate_image_dimensions(jsonl_file)
    print("Width Distribution:")
    plot_histogram_and_stats(widths, 'Width')
    print("\nHeight Distribution:")
    plot_histogram_and_stats(heights, 'Height')

# 更换数据集路径
main("res.jsonl")
