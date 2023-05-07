from examples.example_calculator.calc_conf import (register_node, OP_NODE_ADD,
                                                   OP_NODE_SUB, OP_NODE_MUL,
                                                   OP_NODE_DIV, OP_NODE_BatchNorm2d,
                                                   OP_NODE_CONV2D, OP_NODE_FLATTEN,
                                                   OP_NODE_LINEAR, OP_NODE_MaxPool2d,
                                                   OP_NODE_RELU, OP_NODE_DROPOUT,
                                                   OP_NODE_AdaptiveAvgPool2d, OP_NODE_Softmax,
                                                   OP_NODE_SUM, OP_NODE_Rearrange,
                                                   OP_NODE_Reduce, OP_NODE_Matmul,
                                                   OP_NODE_Sigmoid, OP_NODE_Stack0,
                                                   OP_NODE_Cat0, OP_NODE_AdaptiveMaxPool2d,
                                                   OP_NODE_Cat1, OP_NODE_BatchNorm1d,
                                                   OP_NODE_CONV1D)
from examples.example_calculator.calc_node_base import CalcNode
import torch
from einops.layers.torch import Rearrange, Reduce


@register_node(OP_NODE_ADD)
class CalcNode_Add(CalcNode):
    # icon = "icons/add.png"
    op_code = OP_NODE_ADD
    op_title = "Add"
    content_label = "+"
    content_label_objname = "calc_node_bg"

    def evalOperation(self, input1, input2):
        return input1 + input2


@register_node(OP_NODE_SUB)
class CalcNode_Sub(CalcNode):
    # icon = "icons/sub.png"
    op_code = OP_NODE_SUB
    op_title = "Substract"
    content_label = "-"
    content_label_objname = "calc_node_bg"

    def evalOperation(self, input1, input2):
        return input1 - input2


@register_node(OP_NODE_MUL)
class CalcNode_Mul(CalcNode):
    # icon = "icons/mul.png"
    op_code = OP_NODE_MUL
    op_title = "Multiply"
    content_label = "*"
    content_label_objname = "calc_node_mul"

    def evalOperation(self, input1, input2):
        # print('foo')
        return input1 * input2


@register_node(OP_NODE_DIV)
class CalcNode_Div(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_DIV
    op_title = "Divide"
    content_label = "/"
    content_label_objname = "calc_node_div"

    def evalOperation(self, input1, input2):
        return input1 / input2


# way how to register by function call
# register_node_now(OP_NODE_ADD, CalcNode_Add)

@register_node(OP_NODE_BatchNorm2d)
class CalcNode_BatchNorm2d(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_BatchNorm2d
    op_title = "BatchNorm2d"
    # content_label = "log"
    content_label_objname = "calc_node_BatchNorm2d"

    # def __init__(self, scene):
    #     super().__init__(scene, inputs=[1], outputs=[1])
    #     self.eval()

    def evalOperation(self, input1, input2):
        try:
            return eval('torch.nn.BatchNorm2d'+input2)(input1)
        except:
            return torch.nn.BatchNorm2d(**input2)(input1)
@register_node(OP_NODE_BatchNorm1d)
class CalcNode_BatchNorm1d(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_BatchNorm1d
    op_title = "BatchNorm1d"
    # content_label = "log"
    content_label_objname = "calc_node_BatchNorm1d"

    # def __init__(self, scene):
    #     super().__init__(scene, inputs=[1], outputs=[1])
    #     self.eval()

    def evalOperation(self, input1, input2):
        try:
            return eval('torch.nn.BatchNorm1d'+input2)(input1)
        except:
            return torch.nn.BatchNorm1d(**input2)(input1)

@register_node(OP_NODE_CONV2D)
class CalcNode_CONV2D(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_CONV2D
    op_title = "Conv2d"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_conv2d"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.Conv2d'+input2)(input1)
        except:
            return torch.nn.Conv2d(**input2)(input1)
@register_node(OP_NODE_CONV1D)
class CalcNode_CONV1D(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_CONV1D
    op_title = "Conv1d"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_conv1d"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.Conv1d'+input2)(input1)
        except:
            return torch.nn.Conv1d(**input2)(input1)

@register_node(OP_NODE_FLATTEN)
class CalcNode_FLATTEN(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_FLATTEN
    op_title = "Flatten"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_flatten"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.Flatten'+input2)(input1)
        except:
            return torch.nn.Flatten(**input2)(input1)


@register_node(OP_NODE_LINEAR)
class CalcNode_LINEAR(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_LINEAR
    op_title = "Linear"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_linear"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.Linear'+input2)(input1)
        except:
            return torch.nn.Linear(**input2)(input1)


@register_node(OP_NODE_MaxPool2d)
class CalcNode_MaxPool2d(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_MaxPool2d
    op_title = "MaxPool2d"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_MaxPool2d"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.MaxPool2d'+input2)(input1)
        except:
            return torch.nn.MaxPool2d(**input2)(input1)


@register_node(OP_NODE_RELU)
class CalcNode_RELU(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_RELU
    op_title = "ReLU"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_RELU"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.ReLU'+input2)(input1)
        except:
            return torch.nn.ReLU(**input2)(input1)


@register_node(OP_NODE_DROPOUT)
class CalcNode_DROPOUT(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_DROPOUT
    op_title = "Dropout"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_DROPOUT"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.Dropout'+input2)(input1)
        except:
            return torch.nn.Dropout(**input2)(input1)


@register_node(OP_NODE_AdaptiveAvgPool2d)
class CalcNode_AdaptiveAvgPool2d(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_AdaptiveAvgPool2d
    op_title = "AdaptiveAvgPool2d"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_AdaptiveAvgPool2d"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.AdaptiveAvgPool2d'+input2)(input1)
        except:
            return torch.nn.AdaptiveAvgPool2d(**input2)(input1)

@register_node(OP_NODE_AdaptiveMaxPool2d)
class CalcNode_AdaptiveMaxPool2d(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_AdaptiveMaxPool2d
    op_title = "AdaptiveMaxPool2d"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_AdaptiveMaxPool2d"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.AdaptiveMaxPool2d'+input2)(input1)
        except:
            return torch.nn.AdaptiveMaxPool2d(**input2)(input1)
# torch.nn.AdaptiveMaxPool2d()

@register_node(OP_NODE_Softmax)
class CalcNode_Softmax(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_Softmax
    op_title = "Softmax"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_Softmax"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.Softmax'+input2)(input1)
        except:
            return torch.nn.Softmax(**input2)(input1)


@register_node(OP_NODE_SUM)
class CalcNode_SUM(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_SUM
    op_title = "Sum"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_Sum"

    def evalOperation(self, input1, input2):
        # print(input2)
        return torch.sum(input1, **input2)


@register_node(OP_NODE_Rearrange)
class CalcNode_Rearrange(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_Rearrange
    op_title = "Rearrange"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_Rearrange"

    def evalOperation(self, input1, input2):
        # print(input2)
        # Rearrange(pattern='')
        try:
            return eval('Rearrange'+input2)(input1)
        except:
            return Rearrange(**input2)(input1)

@register_node(OP_NODE_Reduce)
class CalcNode_Rearrange(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_Reduce
    op_title = "Reduce"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_Reduce"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('Reduce'+input2)(input1)
        except:
            return Reduce(**input2)(input1)

@register_node(OP_NODE_Matmul)
class CalcNode_Rearrange(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_Matmul
    op_title = "Matmul"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_Matmul"

    def evalOperation(self, input1, input2):
        # print(input2)

        return torch.matmul(input1, input2)

# torch.nn.Sigmoid
@register_node(OP_NODE_Sigmoid)
class CalcNode_Softmax(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_Sigmoid
    op_title = "Sigmoid"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_Sigmoid"

    def evalOperation(self, input1, input2):
        # print(input2)
        try:
            return eval('torch.nn.Sigmoid'+input2)(input1)
        except:
            return torch.nn.Sigmoid(**input2)(input1)

@register_node(OP_NODE_Stack0)
class CalcNode_Rearrange(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_Stack0
    op_title = "Stack0"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_Stack0"

    def evalOperation(self, input1, input2):
        # print(input2)
        return torch.stack([input1, input2], 0)

@register_node(OP_NODE_Cat0)
class CalcNode_Rearrange(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_Cat0
    op_title = "Cat0"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_Cat0"

    def evalOperation(self, input1, input2):
        # print(input2)
        return torch.cat([input1, input2], 0)

@register_node(OP_NODE_Cat1)
class CalcNode_Rearrange(CalcNode):
    # icon = "icons/divide.png"
    op_code = OP_NODE_Cat1
    op_title = "Cat1"
    # content_label = "Conv2d"
    content_label_objname = "calc_node_Cat1"

    def evalOperation(self, input1, input2):
        # print(input2)
        return torch.cat([input1, input2], 1)

