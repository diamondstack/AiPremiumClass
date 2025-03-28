# 实现基于豆瓣top250图书评论的简单推荐系统（TF-IDF算法实现）
import csv
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(fimename):
    #图书评论信息集合
    book_comments = {}  #key:图书名，value:评论列表

    #打开文件
    with open(fimename,"r",encoding="utf-8") as f:
        #识别格式文本中的标题列
        reader = csv.DictReader(f,delimiter="\t")
        for item in reader:
            book = item["book"]
            comment = item["body"]
            comments_words = jieba.lcut(comment)

            if book == "":  #如果书名为空，则跳过
                continue

            #图书评论集合收集
            book_comments[book] = book_comments.get(book,[])
            book_comments[book].extend(comments_words)

    return book_comments

if __name__ == "__main__":
    #加载停用词列表
    stop_words = [line.strip() for line in open("./2 Language_Model/test/stopwords.txt","r",encoding="utf-8")]

    #加载图书评论数据
    book_comments = load_data("./2 Language_Model/test/doubanbook_top250_comments_fixed.txt")
    print(len(book_comments))
    
    #提取书名和评论文本
    book_names = []
    book_comms = []
    for book,comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)

    #构建TF-IDF矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([" ".join(comms) for comms in book_comms])

    #计算图书之间的余弦相似度
    cosine_matrix = cosine_similarity(tfidf_matrix)

    #输入图书名，输出相似图书
    book_list = list(book_comments.keys())
    print(book_list)
    book_name = input("请输入图书名：")
    book_idx = book_list.index(book_name)   #获取图书索引

    #获取与输入图书最相近的图书
    recommend_books_index = np.argsort(-cosine_matrix[book_idx])[1:11]
    #输出推荐的图书
    print("推荐的图书：")
    for idx in recommend_books_index:
        print(f"《{book_list[idx]}》\t相似度：{cosine_matrix[book_idx][idx]:.4f}")
    print()
