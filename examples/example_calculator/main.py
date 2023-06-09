import os
import sys
from qtpy.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from examples.example_calculator.calc_window import CalculatorWindow


# git push -f legoai master 强制覆盖所有
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # print(QStyleFactory.keys())
    app.setStyle('Fusion')
    # app.setStyle('Windows')

    wnd = CalculatorWindow()
    wnd.setWindowTitle("LegoAI")
    wnd.show()

    sys.exit(app.exec_())
