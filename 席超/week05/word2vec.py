import fasttext
import jieba

# #文档预处理
# with open("./2 Language_Model/test/HLM.txt", "r", encoding="utf-8") as f:
#     lines = f.read()

# with open("./2 Language_Model/test/HML_SPARES.txt", "w", encoding="utf-8") as f:
#     #使用jieba分词器进行分词
#     f.write(" ".join(jieba.cut(lines)))
#训练模型
model = fasttext.train_unsupervised("./2 Language_Model/test/HML_SPARES.txt", model="skipgram")

# print("文档词汇表：",model.words)
print("文档词汇表长度：",len(model.words))

#获取词向量的近邻词
print(model.get_nearest_neighbors("宝玉"))

#获取词向量
print(model.get_word_vector("宝玉"))

#分析词间类比
print(model.get_analogies("宝玉","黛玉","宝钗"))

#保存模型
model.save_model("./2 Language_Model/test/model.bin")

# #加载模型
# model = fasttext.load_model("./2 Language_Model/test/model.bin")
