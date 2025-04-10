import random

# 打开文件用于写入
with open("multiplication_problems.instruct.txt", "w", encoding="utf-8") as file:
    # 生成100个问题
    for i in range(100):
        # 随机生成两个三位数
        num1 = random.randint(100, 999)
        num2 = random.randint(100, 999)
        
        # 计算答案
        answer = num1 * num2
        
        # 创建问题字符串
        question = f"{num1}*{num2}等于多少,请回答答案前先仔细思考计算下，可以把三位整数拆开，然后利用分配律进行计算，提高准确率"
        
        # 将问题和答案写入文件，用tab分隔
        file.write(f"{question}\t{answer}\n")
        
        # 同时打印到控制台
        print(f"{question}\t{answer}")

print("已生成100个三位数乘法问题并保存到multiplication_problems.txt文件中")