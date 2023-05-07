import os
import sys
import torch
# 这行代码是从qtpy的QtWidgets模块中导入QApplication类。
# QApplication类是Qt框架中的核心类之一，用于管理应用程序的事件循环和资源分配。
# 它提供了一个应用程序的主事件循环，用于接收来自操作系统的事件，
# 比如键盘输入、鼠标输入、窗口大小改变等等，并分发给Qt库中的各个组件处理。
# QtWidgets模块则提供了许多常见的UI组件（如窗口、按钮、文本框等）的类。
from qtpy.QtWidgets import QApplication

# 这段代码是将上级目录（即当前脚本的父目录的父目录）添加到系统路径中，
# 以便可以在脚本中导入该上级目录中的模块或包。
#
# 具体解释如下：
#
# -  `os.path.dirname(__file__)`  表示当前脚本的目录路径。
# -  `..`  表示上级目录。
# -  `os.path.join()`  将三部分拼接成一个完整的目录路径。
# -  `sys.path`是Python默认的搜索路径，包括当前目录、安装的第三方包等等，可以通过修改它来增加自定义的搜索路径。
# -  `sys.path.insert(0,  ...)`将新路径添加到`sys.path`的第一个位置，以优先搜索该路径。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from examples.example_calculator.calc_window import CalculatorWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # print(QStyleFactory.keys())
    app.setStyle('Fusion')
    # app.setStyle('Windows')

    wnd = CalculatorWindow()
    wnd.show()

    sys.exit(app.exec_())
