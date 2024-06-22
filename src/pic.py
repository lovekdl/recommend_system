import json
import matplotlib.pyplot as plt

# 读取 JSON 文件
with open('data/dnn1/log.json', 'r') as file:
    data = json.load(file)

# 获取数据
y = data["train_aucs"]

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(y, linestyle='-', color='b', label='Test AUC')

# 设置图表标题和标签
plt.title('Test AUC')
plt.xlabel('step')
plt.ylabel('AUC')
plt.legend()

# 显示图表
plt.grid(True)

plt.savefig('train_aucs_plot.png')

plt.show()
