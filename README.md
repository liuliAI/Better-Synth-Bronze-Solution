# Better Synth 

1. 依赖安装
- 推荐使用 conda 环境
```shell
conda create -n dj python=3.10
conda activate dj

bash install.sh
```

2. 比赛资源下载
- 下载基础模型，种子/微调/评测数据集
- 基础模型与微调数据集均存放于训练目录中指定位置
- 种子数据集存放于`input`目录
- 评测数据集存放于`toolkit/eval`目录
```shell
bash download.sh
```

3. 数据处理与合成
- 比赛要求使用 [data-juicer](https://github.com/modelscope/data-juicer) 基于上一步中下载的**种子数据集**进行数据处理与合成
- `processed_data.jsonl`需为标准的`JSONL`格式，例如：
```json lines
{"images": ["images/00237/002375592.jpg"], "text": "<image>\nadorable pink and gray elephant themed party favour boxes with tissue fillers <|__dj__eoc|>", "id": "002375592"}
{"images": ["images/00199/001999195.jpg"], "text": "<image>\nbreccinano adult dog food for all ages with turkey, lamb and venisi <|__dj__eoc|>", "id": "001999195"}
...
```

4. 执行模型训练/推理
- 模型训练与推理
```shell
cd toolkit/

# 请根据自身需求修改训练脚本train_mgm_2b_stage_1.sh内的参数
# 您只能修改以下范围内的参数

# 修改完毕后执行训练与推理脚本
bash train_mgm_2b_stage_1.sh
```
- 训练与推理结束后，会在`output`目录中产出训练好的模型以及评测集推理结果
  - 训练后的模型存放于：`output/training_dirs`
  - 推理结果存放于：`output/eval_results`

5.思路
<b><font class=center>赛题</font></b>
Better Synth 是一项以数据为中心的挑战赛，考察如何合成与清洗图文数据以在多模态大模型上取得更优的图片理解能力。在给定的种子数据集的基础上，通过高效的数据合成方法与模型生成出更优的数据，并在给定计算量的约束下，实现对图像理解多模态大模型的高效训练。

 <b>1. 数据分析</b>
  - <b>图文匹配相似度</b>
 使用CLIP统计一下数据集中图文匹配的相似度直方图分布。
 ![enter image description here](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/172623128877543621726231288775_hb36njifnu.jpeg)

Maximum Score: 0.4763159155845642

Average Score: 0.3232152164191008

Standard Deviation: 0.03921227440944236

3 sigma 区间: (0.2055783931907737, 0.44085203964742786)

在这里使用K-sigma 进行异常点之外的过滤效果不太理想，把平均值0.3232以下的低质量，图文相关性不大的数据集进行过滤是比较work的。

  - <b>文字长度分析</b>
  简单统计一下文本单词数量直方图分布（由于是英语数据集，我是以空格区分的简单逻辑）
  可以看到，数据集原本caption的总体长度也偏短，因此不进行recapition效果也不会差（由于模型参数量少，Caption越简单越直观越好，不需要图片中其他复杂推理逻辑或其他间接信息）。
  ![enter image description here](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/172623158123115971726231581231_fllwxhmdxh.jpeg)

Maximum word count: 23

Minimum word count: 3

Average word count: 10.79

标准差: 3.52
  
  - <b>图片尺寸分析</b>
  由于存在两个异常数据（一个宽为12192，一个高为3033），把数据范围拉大了，而绝大数图片的尺寸是正常的，导致画出来的图不好看。
  ![enter image description here](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/172623285697282141726232856972_mecmq0gazl.png)
  ![enter image description here](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/172624003140119171726240031401_giauiehhfz.png)

Maximum image width: 12192

Minimum image width: 336

Average image width: 403.1335

![enter image description here](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/172624000503758171726240005037_ac2wj1rqbs.png)

Maximum image height: 3033

Minimum image height: 336

Average image height: 367.7052

由于本次比赛基于[Mini-Gemini](https://github.com/dvlab-research/MGM?spm=a2c22.12281978.0.0.376b2c2bOxjWmv)进行训练，而该模型与训练的像素大小为336，CLIP预训练的像素大小为768，因此我们选择将图片像素小于336大于768的进行去除。
 
 2. 组合算子
 由于核心是模态对齐，文本图片更一致的进行训练效果会更好，因此CLIP必不可少。
 在初赛最后一天使用Stable Diffusion进行图片的重新生成效果并不好。
 使用LLaVA-1.5效果一般，使用prompt：“Please describe the main elements in the image in concise language based on the content. For example, 'A little girl wearing blue clothes is swinging in the park.' Pay attention to maintaining consistency and conciseness in the description, only describe based on the surface content of the image, without delving into the meaning behind the image.”

达到2分，但caption仍然较长，可能是prompt设计的还不够好，使用BLIP-2虽然生成的caption虽然看起来不大行，但训练后效果反而还行。
 值得注意的是，使用BLIP-2+watermark效果更好，这是因为BLIP-2对于有水印的图片生成的caption会在最后包含不必要的信息stock photo![enter image description here](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/172623969726649371726239697266_rvyny5y5cf.jpeg)

使用Aesthetice去除美学评分低的效果不大好，使用NSFW去除暴力、色情图片效果也不大好，Id_score，Grounding_recall，Action num，perplexity score等效果一般

![enter image description here](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/172630001664645271726300016646_7jztugddfz.jpeg)

通过测量不同算子的运行速度（单位：examples/s），获得最佳顺序，具体方案见solution

3. 思路技巧
 - 训练过程并不稳定，一样的设置可能从0.8或者飙升到1.5，因此最好多跑几遍baseline
 -  数据量对模型能力提升有很大帮助，因此清洗筛选完最好进行补充或者重复，达到约束的上限量
 -  高质量重复数据对模型有很大帮助，我们的最佳成绩重复了3.76次（在toolkit/training/preprocess/check_sample_number.py代码进行修改，使用while循环和random.sample进行随机采样，补充至上限）
 - 不同算子顺序导致的结果有一定区别，包括速度也不一样，优先使用轻量级算子更快，但若是考虑模型精度，可以思考先cation再过滤还是先过滤再caption的区别。或者使用 <b>清洗数据→合成数据→清洗数据</b> 的方式
 - 数据多样性很重要