from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import QTextEdit
from examples.example_calculator.calc_conf import register_node, OP_NODE_PARAMS
from examples.example_calculator.calc_node_params_base import CalcNode, CalcGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException


class CalcInputContent(QDMNodeContentWidget):
    def initUI(self):
        pas = str({'in_channels': 3,
                   'out_channels': 16,
                   'kernel_size': 3})
        self.edit = QTextEdit(pas, self)
        # self.edit.setAlignment(Qt.AlignRight)
        self.edit.setFixedSize(155, 105)
        font = QFont()
        font.setPointSize(12)
        self.edit.setFont(font)
        # 设置多行文本框的背景颜色为黑色
        self.edit.setStyleSheet('background-color:#292929;margin-left:6')
        # self.edit.setPlaceholderText("Enter text here")
        # self.edit.setObjectName(self.node.content_label_objname)
        # self.height = 120



    def serialize(self):
        res = super().serialize()
        res['value'] = self.edit.toPlainText()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            value = data['value']
            self.edit.setText(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_PARAMS)
class CalcNode_Input(CalcNode):
    # icon = "icons/in.png"
    op_code = OP_NODE_PARAMS
    op_title = "Params"
    content_label_objname = "calc_node_params"


    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[3])
        self.eval()
        # self.height

    def initInnerClasses(self):
        self.content = CalcInputContent(self)
        self.grNode = CalcGraphicsNode(self)
        self.content.edit.textChanged.connect(self.onInputChanged)

    def evalImplementation(self):
        u_value = self.content.edit.toPlainText()
        s_value = eval(u_value)
        self.value = s_value
        self.markDirty(False)
        self.markInvalid(False)

        self.markDescendantsInvalid(False)
        self.markDescendantsDirty()

        self.grNode.setToolTip("")

        self.evalChildren()

        return self.value