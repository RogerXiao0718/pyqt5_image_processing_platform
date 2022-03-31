from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from thresholding_dialog import Ui_Dialog as Thresholding_dialog
import sys

class MainWindow_controller(QtWidgets.QMainWindow):

   ROISelectionActivated = False

   # 建構子
   def __init__(self):
      super(MainWindow_controller, self).__init__()
      self.ui = Ui_MainWindow()
      self.ui.setupUi(self)
      self.setup_control()
      self.event_binding()


   # 設定元件
   def setup_control(self):
      self.ui.imageDisplayer.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
      self.ui.statusLabel = QtWidgets.QLabel(self.statusBar())
      self.ui.statusLabel.setText("")
      self.statusBar().addPermanentWidget(self.ui.statusLabel)
      self.ui.thresholding_dialog = Thresholding_dialog()
      self.ui.thresholding_dialog.thresholding_label.setText(f"{self.ui.thresholding_dialog.thresholding_slider.value()}")
      self.binaryThreshold = self.ui.thresholding_dialog.thresholding_slider.value()

   # 負責綁定事件
   def event_binding(self):
      self.ui.actionOpen_Image.triggered.connect(self.loadImageClicked)
      self.ui.actionSave_Image.triggered.connect(self.saveImageClicked)
      self.ui.action_ROI.triggered.connect(self.setROIClicked)
      self.ui.imageDisplayer.mousePressEvent = self.pressROISelection
      self.ui.imageDisplayer.mouseReleaseEvent = self.releaseROISelection
      self.ui.actionRestoreOriginal.triggered.connect(self.restoreOriginalClicked)
      self.ui.action_showHisto.triggered.connect(self.showHistogram)
      self.ui.action_convertGrayScale.triggered.connect(self.convertGrayscaleClicked)
      self.ui.action_thresholding.triggered.connect(self.thresholdingClicked)
      self.ui.thresholding_dialog.thresholding_slider.valueChanged.connect(self.update_binary_threshold)
      self.ui.action_histoEqualization.triggered.connect(self.histoEqualizationClicked)


   def imageDisplay(self, image):
      displayer_size = self.ui.imageDisplayer.size()
      self.displayed_image = cv2.resize(image, (displayer_size.width(), displayer_size.height()),
                                        interpolation=cv2.INTER_AREA)

      # 透過判斷式判斷是否為彩色或灰階
      if len(self.displayed_image.shape) == 3:
         height, width, channel = self.displayed_image.shape
         bytesPerLine = width * channel
         self.qimage = QImage(self.displayed_image, width, height, bytesPerLine, QImage.Format_BGR888)
         self.ui.imageDisplayer.setPixmap(QPixmap.fromImage(self.qimage))
      elif len(self.displayed_image.shape) == 2:
         height, width = self.displayed_image.shape
         bytesPerLine = width * 1
         self.qimage = QImage(self.displayed_image, width, height, bytesPerLine, QImage.Format_Grayscale8)
         self.ui.imageDisplayer.setPixmap(QPixmap.fromImage(self.qimage))

   def loadImageClicked(self):
      filename, filetype = QFileDialog.getOpenFileName(self, "Open Image", "./")
      if filename:
         #為了相容中文路徑，使用numpy的fromfile並用讀取資料後透過cv2將資料decode
         self.cv2_image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
         self.original_image = self.cv2_image.copy()
         self.ui.statusLabel.setText(f"Load image from {filename}")
         self.imageDisplay(self.cv2_image)


   # 儲存圖片
   def saveImageClicked(self):
      filename, filetype = QFileDialog.getSaveFileName(self, "Save Image", "./")
      if filename and (self.cv2_image is not None):
         fileExt = os.path.splitext(filename)[1]
         # 為了相容中文路徑，使用cv2編碼後透過numpy的tofile儲存圖檔
         cv2.imencode(fileExt, self.cv2_image)[1].tofile(filename)


   def setROIClicked(self):
      self.ROISelectionActivated = True


   # ROI起點的選取
   def pressROISelection(self, event):
      if self.ROISelectionActivated:
         self.ROISelection_startPos = event.localPos()


   # 設定ROI終點，並透過比大小決定左上與右下座標，透過numpy的範圍取值來更新圖片
   def releaseROISelection(self, event):
      if self.ROISelectionActivated:
         self.ROISelection_endPos = event.localPos()
         start_x, start_y = int(self.ROISelection_startPos.x()), int(self.ROISelection_startPos.y())
         end_x, end_y = int(self.ROISelection_endPos.x()), int(self.ROISelection_endPos.y())
         ltx, lty = min(start_x, end_x), min(start_y, end_y)
         rbx, rby = max(start_x, end_x), max(start_y, end_y)
         self.displayed_image = self.displayed_image[lty:rby+1, ltx:rbx+1]
         self.imageDisplay(self.displayed_image)
         self.ui.statusLabel.setText(f"From: ({ltx}, {lty})  To: ({rbx}, {rby})")
         self.ROISelectionActivated = False


   # 恢復原始圖片的事件
   def restoreOriginalClicked(self):
      self.restoreOriginal()


   def restoreOriginal(self):
      self.cv2_image = self.original_image.copy()
      self.imageDisplay(self.cv2_image)


   # 顯示影像的直方圖，若為彩色影像則顯示bgr的直方圖，若為灰階則顯示灰階的直方圖
   def showHistogram(self):
      if len(self.cv2_image.shape) == 3:
         color = ('b', 'g', 'r')
         alpha = (0.6, 0.6, 0.5)
         for i, col in enumerate(color):
            histr = cv2.calcHist([self.cv2_image], [i], None, [256], [0, 256])
            plt.bar(range(0, 256), histr.ravel(), color=color[i], alpha=alpha[i])
      elif len(self.cv2_image.shape) == 2:
         color = 'gray'
         alpha = 0.8
         histr = cv2.calcHist([self.cv2_image], [0], None, [256], [0, 256])
         plt.bar(range(0, 256), histr.ravel(), color=color, alpha=alpha)


      plt.title("Histogram")
      plt.show()


   # 轉換圖片為灰階
   def convertGrayscaleClicked(self):
      self.convertGrayscale()
      self.ui.statusLabel.setText("轉換灰階")


   def convertGrayscale(self):
      if len(self.cv2_image.shape) == 3:
         self.cv2_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2GRAY)
         self.imageDisplay(self.cv2_image)


   # 若二值化閥值的Slider數值更動則更新binaryThreshold
   def update_binary_threshold(self):
      self.binaryThreshold = self.ui.thresholding_dialog.thresholding_slider.value()
      self.ui.thresholding_dialog.thresholding_label.setText(f"{self.binaryThreshold}")


   # 處理二值化
   def thresholdingClicked(self):
      returnValue = self.ui.thresholding_dialog.exec_()
      if returnValue == QtWidgets.QDialog.Accepted:
         self.imageThresholding(self.binaryThreshold)

   # 透過傳入的threshold進行圖片二值化
   def imageThresholding(self, threshold):
      self.restoreOriginal()
      self.convertGrayscale()
      self.cv2_image = cv2.threshold(self.cv2_image, threshold, 255, cv2.THRESH_BINARY)[1]
      self.imageDisplay(self.cv2_image)
      self.ui.statusLabel.setText(f"影像二值化 Threshold: {self.binaryThreshold}")


   # 處理直方圖等化
   def histoEqualizationClicked(self):
      self.restoreOriginal()
      self.histoEqualization()
      self.ui.statusLabel.setText("直方圖等化");

   def histoEqualization(self):
      self.convertGrayscale()
      self.cv2_image = cv2.equalizeHist(self.cv2_image)
      self.imageDisplay(self.cv2_image)
