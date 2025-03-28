import fasttext

# 训练模型
model = fasttext.train_supervised("./2 Language_Model/test/cooking.stackexchange.txt",epoch=10,dim=200)

#文本分类
print(model.predict("Which baking dish is best to bake a banana bread?"))

print(model.predict("Is it safe to eat food that was heated in plastic wrap to the point the plastic wrap flamed?"))
