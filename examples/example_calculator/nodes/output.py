from qtpy.QtWidgets import QLabel
from qtpy.QtCore import Qt
from examples.example_calculator.calc_conf import register_node, OP_NODE_OUTPUT
from examples.example_calculator.calc_node_base import CalcNode, CalcGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget


class CalcOutputContent(QDMNodeContentWidget):
    def initUI(self):
        self.lbl = QLabel("100", self)
        self.lbl.setAlignment(Qt.AlignLeft)
        self.lbl.setFixedWidth(160)
        self.lbl.setFixedHeight(40)
        self.lbl.setStyleSheet("QLabel {font-size: 18px; margin-top: 15px; margin-left:12px; radius:10px}")


        # self.lbl = QLabel("42", self)
        # self.lbl.setAlignment(Qt.AlignLeft)
        # self.lbl.setObjectName(self.node.content_label_objname)


@register_node(OP_NODE_OUTPUT)
class CalcNode_Output(CalcNode):
    # icon = "icons/out.png"
    op_code = OP_NODE_OUTPUT
    op_title = "Output"
    content_label_objname = "calc_node_output"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[])

    def initInnerClasses(self):
        self.content = CalcOutputContent(self)
        self.grNode = CalcGraphicsNode(self)

    def evalImplementation(self):
        input_node = self.getInput(0)
        if not input_node:
            self.grNode.setToolTip("Input is not connected")
            self.markInvalid()
            return

        val = input_node.eval()

        if val is None:
            self.grNode.setToolTip("Input is NaN")
            self.markInvalid()
            return
        # print(val.shape)
        self.content.lbl.setText(str(val.shape).split('(')[1][1:-2])
        self.markInvalid(False)
        self.markDirty(False)
        self.grNode.setToolTip("")

        return val
