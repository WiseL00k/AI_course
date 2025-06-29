from cnn_mnist import CNN
from torchview import draw_graph

model = CNN()
batch_size = 32
graph = draw_graph(model, input_size=(batch_size, 1, 28, 28))
graph.visual_graph.render(format='png', filename='cnn_architecture')
