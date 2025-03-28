#修复后文件存盘
fixed = open("./2 Language_Model/test/doubanbook_top250_comments_fixed.txt", "w", encoding="utf-8")

lines = [line for line in open("./2 Language_Model/test/doubanbook_top250_comments.txt", "r", encoding="utf-8")]


for i,line in enumerate(lines):
    if i == 0:
        fixed.write(line)
        prev_line = ""  #上一行书名置为空
        continue
    #提取书名和评论文本
    terms = line.split("\t")

    #如果书名和上一行书名相同，则说明是上一行的评论
    if terms[0] == prev_line.split("\t")[0]:
        if len(prev_line.split("\t")) == 6: #上一行是评论
            #保存上一行的记录
            fixed.write(prev_line + "\n")
            prev_line = line.strip()    #保存当前行
        else:
            prev_line  = ""
    else:
        if len(terms) == 6:     #新书评论
            prev_line = line.strip()
        else:
            prev_line += line.strip()  #合并当前行和上一行的评论

fixed.close()






    
