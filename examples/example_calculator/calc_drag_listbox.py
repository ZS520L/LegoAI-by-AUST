from PyQt5.QtWidgets import QTreeWidgetItem
from qtpy.QtGui import QPixmap, QIcon, QDrag
from qtpy.QtCore import QSize, Qt, QByteArray, QDataStream, QMimeData, QIODevice, QPoint
from qtpy.QtWidgets import QTreeWidget, QAbstractItemView, QListWidgetItem

from examples.example_calculator.calc_conf import CALC_NODES, get_class_from_opcode, LISTBOX_MIMETYPE
from nodeeditor.utils import dumpException


class QDMDragListbox(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        # init
        self.setIconSize(QSize(32, 32))
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setDragEnabled(True)

        self.setHeaderHidden(True)  # 隐藏表头
        self.setAnimated(True)  # 启用动画
        # Set the text color to white
        self.setStyleSheet("QTreeWidget { color: white; font-size: 16px;}")

        self.addMyItems()

    def addMyItems(self):
        parent_nodes = {
            "In_Out": [],
            "Layers": [],
            "Math_Operate": []
        }

        keys = list(CALC_NODES.keys())
        keys.sort()

        for key in keys:
            node = get_class_from_opcode(key)

            if key < 4:
                parent_nodes["In_Out"].append(node)
            elif key < 130:
                parent_nodes["Layers"].append(node)
            else:
                parent_nodes["Math_Operate"].append(node)

        for parent_name, child_nodes in parent_nodes.items():
            parent_item = QTreeWidgetItem(self)
            parent_item.setText(0, parent_name)
            parent_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

            for node in child_nodes:
                self.addMyItem(parent_item, node.op_title, node.icon, node.op_code)

    def addMyItem(self, parent, name, icon=None, op_code=0):
        item = QTreeWidgetItem(parent)
        pixmap = QPixmap(icon if icon is not None else ".")
        item.setIcon(0, QIcon(pixmap))

        item.setText(0, name)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)

        # setup data
        item.setData(0, Qt.UserRole, pixmap)
        item.setData(0, Qt.UserRole + 1, op_code)

    def startDrag(self, *args, **kwargs):
        try:
            item = self.currentItem()

            # Check if the item has children (categories/parent nodes)
            if item.childCount() > 0:
                return

            op_code = item.data(0, Qt.UserRole + 1)
            pixmap = QPixmap(item.data(0, Qt.UserRole))

            itemData = QByteArray()
            dataStream = QDataStream(itemData, QIODevice.WriteOnly)
            dataStream << pixmap
            dataStream.writeInt(op_code)
            dataStream.writeQString(item.text(0))

            mimeData = QMimeData()
            mimeData.setData(LISTBOX_MIMETYPE, itemData)

            drag = QDrag(self)
            drag.setMimeData(mimeData)
            drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
            drag.setPixmap(pixmap)

            drag.exec_(Qt.MoveAction)

        except Exception as e:
            dumpException(e)
