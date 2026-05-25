import matplotlib.pyplot as plt

# 数据准备
groups = ['A', 'B', 'C', 'D', 'E']
accuracy = [58, 61, 63, 67, 69]

# 创建图表
plt.figure(figsize=(8, 5))
# 设置柱子颜色为天蓝色，边框为黑色
bars = plt.bar(groups, accuracy, color='#87CEEB', edgecolor='black', width=0.6)

# 设置标题和标签
plt.title('Accuracy by Group', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)

# 设置Y轴范围，让差异看起来更明显一点（从0到80）
plt.ylim(0, 80)

# 在每个柱子上方添加具体的数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加虚线网格，增加可读性
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 显示图表
plt.tight_layout()
plt.show()