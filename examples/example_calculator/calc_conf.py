LISTBOX_MIMETYPE = "application/x-item"

OP_NODE_PARAMS = 0
OP_NODE_INPUT = 1
OP_NODE_OUTPUT = 2
OP_NODE_Value = 3
OP_NODE_ADD = 133
OP_NODE_SUB = 134
OP_NODE_MUL = 135
OP_NODE_DIV = 136
OP_NODE_BatchNorm2d = 7
OP_NODE_CONV2D = 8
OP_NODE_FLATTEN = 9
OP_NODE_LINEAR, OP_NODE_MaxPool2d = 10, 11
OP_NODE_RELU = 12
OP_NODE_DROPOUT = 13
OP_NODE_AdaptiveAvgPool2d = 14
OP_NODE_Softmax = 15
OP_NODE_Sigmoid = 16
OP_NODE_AdaptiveMaxPool2d = 17
OP_NODE_BatchNorm1d = 18
OP_NODE_CONV1D = 19


OP_NODE_SUM = 137
OP_NODE_Rearrange = 138
OP_NODE_Reduce = 139
OP_NODE_Matmul = 140
OP_NODE_Stack0 = 141
OP_NODE_Cat0 = 142
OP_NODE_Cat1 = 143

CALC_NODES = {
}


class ConfException(Exception): pass
class InvalidNodeRegistration(ConfException): pass
class OpCodeNotRegistered(ConfException): pass


def register_node_now(op_code, class_reference):
    if op_code in CALC_NODES:
        raise InvalidNodeRegistration("Duplicate node registration of '%s'. There is already %s" %(
            op_code, CALC_NODES[op_code]
        ))
    CALC_NODES[op_code] = class_reference


def register_node(op_code):
    def decorator(original_class):
        register_node_now(op_code, original_class)
        return original_class
    return decorator

def get_class_from_opcode(op_code):
    if op_code not in CALC_NODES: raise OpCodeNotRegistered("OpCode '%d' is not registered" % op_code)
    return CALC_NODES[op_code]



# import all nodes and register them
from examples.example_calculator.nodes import *