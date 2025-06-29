import sys
import torch
import numpy as np
from main import Net
from cnn_mnist import CNN
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                            QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QGroupBox)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap, QFont

class DrawingCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(256, 256)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.processed_image = None
        self.drawing = False
        self.last_point = None
        self.pen = QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_point:
            painter = QPainter(self.image)
            painter.setPen(self.pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def clear_canvas(self):
        self.image.fill(Qt.white)
        self.update()

    def get_image_array(self):
        img = self.image.scaled(28, 28, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(img.height(), img.width(), 4)[:, :, :3]
        gray = np.dot(arr, [0.2989, 0.5870, 0.1140])
        # 反转颜色：白底黑字 -> 黑底白字
        gray = 1.0 - (gray / 255.0)
        self.processed_image = QImage(img).convertToFormat(QImage.Format_Grayscale8).invertPixels()
        return (gray - 0.1307) / 0.3081

    def get_processed_preview(self):
        """直接返回处理后的28x28灰度图像（黑底白字）"""
        if self.processed_image and not self.processed_image.isNull():
            return QPixmap.fromImage(self.processed_image)
        # 空画布时返回纯黑图像
        return QPixmap(28, 28)

class PredictWorker(QThread):
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, model, image_data):
        super().__init__()
        self.model = model
        self.image_data = image_data

    def run(self):
        try:
            if isinstance(self.model, CNN):
                # CNN需要4D输入 (batch, channel, height, width)
                tensor = torch.tensor(self.image_data, dtype=torch.float32).reshape(1, 1, 28, 28)
            else:
                # 全连接网络需要扁平化输入
                tensor = torch.tensor(self.image_data, dtype=torch.float32).flatten().unsqueeze(0)
            with torch.no_grad():
                output = self.model(tensor)
                prediction = torch.argmax(output).item()
            self.finished.emit(prediction)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model = self.load_model()
        self.predict_thread = None

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)

        # 左侧控制面板
        control_panel = QVBoxLayout()
        
        # 模型选择
        self.model_combo = QComboBox()
        self.model_combo.addItem("全连接网络")
        self.model_combo.addItem("卷积网络")
        control_panel.addWidget(QLabel("选择模型:"))
        control_panel.addWidget(self.model_combo)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        self.result_label = QLabel("预测结果：")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px;")
        
        recognize_btn = QPushButton("识别")
        recognize_btn.clicked.connect(self.start_prediction)
        recognize_btn.setFixedSize(100, 40)
        
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self.clear_all)
        clear_btn.setFixedSize(100, 40)

        # 右侧布局
        right_panel = QVBoxLayout()
        
        # 绘图区域
        self.canvas = DrawingCanvas()
        
        # 处理后的图像预览
        preview_box = QGroupBox("处理后的输入 (28x28)")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(100, 100)
        self.preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        preview_box.setLayout(preview_layout)

        right_panel.addWidget(self.canvas)
        right_panel.addWidget(preview_box)

        control_panel.addWidget(self.result_label)
        control_panel.addWidget(recognize_btn)
        control_panel.addWidget(clear_btn)
        control_panel.addStretch()

        layout.addLayout(control_panel, 1)
        layout.addLayout(right_panel, 2)
        
        self.setCentralWidget(main_widget)
        self.setWindowTitle("手写数字识别")
        self.setFixedSize(800, 500)

    def load_model(self):
        try:
            if self.model_combo.currentText() == "全连接网络":
                model = Net()
                model.load_state_dict(torch.load('model.pth', map_location='cpu'))
            else:
                model = CNN()
                model.load_state_dict(torch.load('cnn_model.pth', map_location='cpu'))
            model.eval()
            return model
        except FileNotFoundError as e:
            print(f"模型加载失败: {str(e)}")
            return None
        except Exception as e:
            print(f"发生错误: {str(e)}")
            return None

    def on_model_changed(self):
        self.model = self.load_model()

    def start_prediction(self):
        if self.model is None or (self.predict_thread and self.predict_thread.isRunning()):
            return
            
        try:
            image_data = self.canvas.get_image_array()
            # 直接显示处理后的原始图像（已包含颜色反转）
            raw_preview = self.canvas.get_processed_preview()
            self.preview_label.setPixmap(raw_preview.scaled(
                100, 100,
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            ))
        except Exception as e:
            self.result_label.setText(f"图像处理错误：{str(e)}")
            return
        
        self.predict_thread = PredictWorker(self.model, image_data)
        self.predict_thread.finished.connect(self.show_prediction_result)
        self.predict_thread.error.connect(self.show_error)
        self.predict_thread.start()

    def show_prediction_result(self, prediction):
        self.result_label.setText(f"预测结果：{prediction}")

    def show_error(self, message):
        self.result_label.setText(f"错误：{message}")

    def clear_all(self):
        self.canvas.clear_canvas()
        self.preview_label.clear()
        self.result_label.setText("预测结果：")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
