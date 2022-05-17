import math
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from thresholding_dialog import Ui_Dialog as Thresholding_dialog
from warpAffine_dialog import Ui_Dialog as WarpAffine_dialog
import random
from utils.erosion import erosion
from utils.morph_shape import morph_shape
from utils.createMorphologyWindow import createMorphWindow
from utils.dilation import dilation
from utils.affineTransform import affineTransform
from utils.perspectiveTransform import perspectiveTransform

random.seed(7414)


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
      self.ui.warpAffine_dialog = WarpAffine_dialog()  # 初始化warpAffineDialog
      self.rotateValue = 0
      self.flip = False
      self.translateXValue = 0
      self.translateYValue = 0
      self.thresholding_value = self.ui.thresholding_dialog.thresholding_slider.value()
      self.perspectiveTransform_counter = -1

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
      self.ui.action_convertHSV.triggered.connect(self.convertHSVClicked)
      self.ui.action_thresholding.triggered.connect(self.thresholdingClicked)
      self.ui.thresholding_dialog.thresholding_slider.valueChanged.connect(self.update_binary_threshold)
      self.ui.action_histoEqualization.triggered.connect(self.histoEqualizationClicked)
      self.ui.action_convertRGB.triggered.connect(self.convertRGBClicked)
      self.ui.action_gaussianFiltering.triggered.connect(self.gaussianFilteringClicked)
      self.ui.action_medianFiltering.triggered.connect(self.medianFilteringClicked)
      self.ui.action_averageFiltering.triggered.connect(self.averageFilteringClicked)
      self.ui.action_boundryDetection.triggered.connect(self.cannyBoundryDetectionClicked)
      self.ui.action_perspectiveTransform.triggered.connect(self.perspectiveTransformClicked)
      self.ui.action_addNoise.triggered.connect(self.gaussianNoiseClicked)
      self.ui.action_bilateralFiltering.triggered.connect(self.bilateralFilteringClick)
      self.ui.action_gaussianHighPass.triggered.connect(self.gaussianHighPassFilteringClicked)
      self.ui.action_laplacianFiltering.triggered.connect(self.laplacianFiltering)
      self.ui.action_affineTransform.triggered.connect(self.affineTransformClicked)
      self.ui.warpAffine_dialog.flipCheckBox.toggled.connect(self.updateFlipValue)
      self.ui.warpAffine_dialog.rotateSlider.valueChanged.connect(self.update_rotateValue)
      self.ui.warpAffine_dialog.translateXSlider.valueChanged.connect(self.update_translateXValue)
      self.ui.warpAffine_dialog.translateYSlider.valueChanged.connect(self.update_translateYValue)
      self.ui.action_CornerHarris.triggered.connect(self.cornerHarrisClicked)
      self.ui.action_FindContours.triggered.connect(self.findContoursClicked)
      self.ui.action_Erosion.triggered.connect(self.erosionClicked)
      self.ui.action_Dilation.triggered.connect(self.dilationClicked)


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
      # For ROISelection
      if self.ROISelectionActivated:
         self.ROISelection_endPos = event.localPos()
         start_x, start_y = int(self.ROISelection_startPos.x()), int(self.ROISelection_startPos.y())
         end_x, end_y = int(self.ROISelection_endPos.x()), int(self.ROISelection_endPos.y())
         ltx, lty = min(start_x, end_x), min(start_y, end_y)
         rbx, rby = max(start_x, end_x), max(start_y, end_y)
         roi_image = self.displayed_image[lty:rby+1, ltx:rbx+1]
         roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
         plt.imshow(roi_image)
         plt.axis('off')
         plt.show()
         self.ui.statusLabel.setText(f"From: ({ltx}, {lty})  To: ({rbx}, {rby})")
         self.ROISelectionActivated = False
      # For perspectiveTransform
      if self.perspectiveTransform_counter > 0:
         localPos = event.localPos()
         x, y = localPos.x(), localPos.y()
         self.perspectiveTransform_pts_dst.append([x, y])
         self.perspectiveTransform_counter -= 1
         if self.perspectiveTransform_counter == 0:
            transformed_image = perspectiveTransform(self.cv2_image, self.perspectiveTransform_pst_src, self.perspectiveTransform_pts_dst)
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            plt.imshow(transformed_image)
            plt.axis('off')
            plt.show()


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


   def convertHSVClicked(self):
      self.convertHSV()
      self.ui.statusLabel.setText("轉換HSV")


   def convertHSV(self):
      self.restoreOriginal()
      self.cv2_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2HSV)
      self.imageDisplay(self.cv2_image)

   def convertRGBClicked(self):
      self.convertRGB()
      self.ui.statusLabel.setText("轉換RGB")

   def convertRGB(self):
      self.restoreOriginal()
      if len(self.cv2_image.shape) == 2:
         self.cv2_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_GRAY2BGR)
         self.imageDisplay(self.cv2_image)


   # 若二值化閥值的Slider數值更動則更新binaryThreshold
   def update_binary_threshold(self):
      self.thresholding_value = self.ui.thresholding_dialog.thresholding_slider.value()
      self.ui.thresholding_dialog.thresholding_label.setText(f"{self.thresholding_value}")


   def update_rotateValue(self):
      self.rotateValue = self.ui.warpAffine_dialog.rotateSlider.value()
      self.ui.warpAffine_dialog.rotateValueLabel.setText(f"{self.rotateValue}")


   def update_translateXValue(self):
      self.translateXValue = self.ui.warpAffine_dialog.translateXSlider.value()
      self.ui.warpAffine_dialog.translateXValueLabel.setText(f"{self.translateXValue}")


   def update_translateYValue(self):
      self.translateYValue = self.ui.warpAffine_dialog.translateYSlider.value()
      self.ui.warpAffine_dialog.translateYValueLabel.setText(f"{self.translateYValue}")


   def updateFlipValue(self):
      self.flip = self.ui.warpAffine_dialog.flipCheckBox.isChecked()


   # 處理二值化
   def thresholdingClicked(self):
      returnValue = self.ui.thresholding_dialog.exec_()
      if returnValue == QtWidgets.QDialog.Accepted:
         self.imageThresholding(self.thresholding_value)

   # 透過傳入的threshold進行圖片二值化
   def imageThresholding(self, threshold):
      self.restoreOriginal()
      self.convertGrayscale()
      self.cv2_image = cv2.threshold(self.cv2_image, threshold, 255, cv2.THRESH_BINARY)[1]
      self.imageDisplay(self.cv2_image)
      self.ui.statusLabel.setText(f"影像二值化 Threshold: {self.thresholding_value}")


   # 處理直方圖等化
   def histoEqualizationClicked(self):
      self.restoreOriginal()
      self.histoEqualization()
      self.ui.statusLabel.setText("直方圖等化");

   def histoEqualization(self):
      self.convertGrayscale()
      self.cv2_image = cv2.equalizeHist(self.cv2_image)
      self.imageDisplay(self.cv2_image)


   def gaussianFilteringClicked(self):
      self.gaussianFiltering()
      self.imageDisplay(self.cv2_image)
      self.ui.statusLabel.setText("高斯濾波")


   def gaussianFiltering(self):
      self.cv2_image = cv2.GaussianBlur(self.cv2_image, (9, 9), 0)


   def medianFilteringClicked(self):
      self.cv2_image = cv2.medianBlur(self.cv2_image, 9)
      self.imageDisplay(self.cv2_image)
      self.ui.statusLabel.setText("中值濾波")


   def averageFilteringClicked(self):
      self.cv2_image = cv2.blur(self.cv2_image, (9, 9))
      self.imageDisplay(self.cv2_image)
      self.ui.statusLabel.setText("均值濾波")


   def bilateralFilteringClick(self):
      kernel_size = 9
      sigma = 100
      self.cv2_image = cv2.bilateralFilter(self.cv2_image, kernel_size, sigma, sigma)
      self.imageDisplay(self.cv2_image)
      self.ui.statusLabel.setText("雙邊濾波")


   def gaussianHighPassFilteringClicked(self):
      kernel_size = 9
      sigma = 100
      self.cv2_image = self.cv2_image - cv2.GaussianBlur(self.cv2_image, (kernel_size, kernel_size), sigma) + 127
      self.imageDisplay(self.cv2_image)
      self.ui.statusLabel.setText("高斯高通濾波")


   def laplacianFiltering(self):
      kernel_size = 9
      self.cv2_image = cv2.Laplacian(self.cv2_image, -1, kernel_size)
      self.imageDisplay(self.cv2_image)
      self.ui.statusLabel.setText("拉普拉斯濾波")


   def cannyBoundryDetectionClicked(self):
      returnValue = self.ui.thresholding_dialog.exec_()
      if returnValue == QtWidgets.QDialog.Accepted:
         self.cannyBoundryDetection(self.thresholding_value)
         self.imageDisplay(self.cv2_image)
         self.ui.statusLabel.setText("Canny邊緣偵測")


   def cannyBoundryDetection(self, max_threshold):
      self.convertGrayscale()
      self.gaussianFiltering()
      self.cv2_image = cv2.Canny(self.cv2_image, 30, max_threshold)


   def perspectiveTransformClicked(self):
      h = self.displayed_image.shape[0]
      w = self.displayed_image.shape[1]
      self.perspectiveTransform_pst_src = np.array([
         [0, 0],
         [w - 1, 0],
         [0, h - 1],
         [w - 1, h - 1]
      ], dtype=np.float32)
      self.perspectiveTransform_pts_dst = []
      self.perspectiveTransform_counter = 4


   def gaussianNoiseClicked(self, mean=0, sigma=0.5):
      normalized_image = self.cv2_image / 255
      noise = np.random.normal(mean, sigma, normalized_image.shape)

      gaussian_out = normalized_image + noise
      gaussian_out = np.clip(gaussian_out, 0, 1)

      gaussian_out = np.uint8(gaussian_out * 255)
      self.cv2_image = gaussian_out
      self.imageDisplay(self.cv2_image)


   def affineTransformClicked(self):
      returnValue = self.ui.warpAffine_dialog.exec_()
      if returnValue == QtWidgets.QDialog.Accepted:
         affineTransform(self.cv2_image, self.rotateValue, self.translateXValue, self.translateYValue, self.flip)



   def cornerHarrisClicked(self):
      gray = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2GRAY)
      gray = np.float32(gray)
      dst = cv2.cornerHarris(gray, 2, 3, 0.04)
      # result is dilated for marking the corners, not important
      dst = cv2.dilate(dst, None)
      # Threshold for an optimal value, it may vary depending on the image.
      self.cv2_image[dst > 0.01 * dst.max()] = [0, 0, 255]
      self.imageDisplay(self.cv2_image)


   def findContoursClicked(self):
      def thresh_callback(val):
         threshold = val
         # Detect edges using Canny
         canny_output = cv2.Canny(target_image, threshold, threshold * 2)
         # Find contours
         contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         # Draw contours
         drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
         for i in range(len(contours)):
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
         # Show in a window
         self.imageDisplay(drawing)
      # Convert image to gray and blur it
      target_image = self.displayed_image.copy()
      gray_image = cv2.cvtColor(self.displayed_image, cv2.COLOR_BGR2GRAY)
      gray_image = cv2.blur(gray_image, (3, 3))
      # Create Window
      source_window = 'Source'
      cv2.namedWindow(source_window)
      cv2.imshow(source_window, gray_image)
      max_thresh = 255
      thresh = 125  # initial threshold
      cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
      thresh_callback(thresh)



   def erosionClicked(self):
      title_trackbar_element_shape = 'Element'
      title_trackbar_kernel_size = 'Kernel size'
      title_erosion_window = 'Erosion'
      def erosionCallback(val):
         src = self.displayed_image
         erosion_size = cv2.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
         shape = morph_shape(cv2.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))
         dst = erosion(src, erosion_size, shape)
         cv2.imshow(title_erosion_window, dst)

      createMorphWindow(title_erosion_window, title_trackbar_element_shape, title_trackbar_kernel_size, erosionCallback)
      cv2.imshow(title_erosion_window, self.displayed_image)

   def dilationClicked(self):
      title_trackbar_element_shape = 'Element'
      title_trackbar_kernel_size = 'Kernel size'
      title_dilation_window = 'Dilation'
      def dilationCallback(val):
         src = self.displayed_image
         erosion_size = cv2.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
         shape = morph_shape(cv2.getTrackbarPos(title_trackbar_element_shape, title_dilation_window))
         dst = dilation(src, erosion_size, shape)
         cv2.imshow(title_dilation_window, dst)

      createMorphWindow(title_dilation_window, title_trackbar_element_shape, title_trackbar_kernel_size, dilationCallback)
      cv2.imshow(title_dilation_window, self.displayed_image)