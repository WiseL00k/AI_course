from main import Net
from cnn_mnist import CNN
import torch
from torchview import draw_graph

model = CNN()
batch_size = 32
graph = draw_graph(model, input_size=(batch_size, 1, 28, 28))
graph.visual_graph.render(format='png', filename='cnn_architecture')

# 创建模型实例
model = Net()

# 可视化模型
model_graph = draw_graph(
    model,
    input_size=(32, 784),       # 批大小32, 输入维度784
    device='cpu',
    show_shapes=True,           # ✔️ 显示维度
    hide_module_functions=False, # ✔️ 显示层名称（关键参数）
    expand_nested=True,
    graph_attributes={'fontname': 'Arial'}  # 避免字体警告
)

# 保存为高分辨率PNG
model_graph.visual_graph.render('fc_network', format='png', cleanup=True)
print("可视化图已保存为 fc_network.png")