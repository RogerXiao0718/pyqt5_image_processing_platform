# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'thresholding_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(QtWidgets.QDialog):

    def __init__(self):
        super(Ui_Dialog, self).__init__()
        self.setupUi(self)

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 351, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.thresholding_slider = QtWidgets.QSlider(Dialog)
        self.thresholding_slider.setGeometry(QtCore.QRect(40, 130, 241, 22))
        self.thresholding_slider.setMaximum(255)
        self.thresholding_slider.setProperty("value", 127)
        self.thresholding_slider.setOrientation(QtCore.Qt.Horizontal)
        self.thresholding_slider.setObjectName("thresholding_slider")
        self.thresholding_label = QtWidgets.QLabel(Dialog)
        self.thresholding_label.setGeometry(QtCore.QRect(310, 130, 58, 15))
        self.thresholding_label.setText("")
        self.thresholding_label.setObjectName("thresholding_label")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
