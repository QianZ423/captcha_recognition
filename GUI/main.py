# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import QApplication
from gui.MainWidget import MainWidget
import sys


def main():

    # 创建QT
    app = QApplication(sys.argv)

    # 声明窗口
    main_widget = MainWidget()

    # 显示窗口
    main_widget.show()

    # 关闭QT
    exit(app.exec_())


    pass


if __name__ == '__main__':
    main()











