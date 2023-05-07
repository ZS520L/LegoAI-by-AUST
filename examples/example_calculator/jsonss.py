import json
import torch
import torch.nn as nn
from lastfun import compose


def nodes2net(data):
    nodes = data['nodes']
    edges = data['edges']

    # 获得所有的连线关系
    edge_dict = {}
    for edge in edges:
        if edge_dict.get(str(edge['start'])) is None:
            edge_dict[str(edge['start'])] = [str(edge['end'])]
        else:
            edge_dict[str(edge['start'])].append(str(edge['end']))

    for edge in edges:
        if edge_dict.get(str(edge['end'])) is None:
            edge_dict[str(edge['end'])] = [str(edge['start'])]
        else:
            edge_dict[str(edge['end'])].append(str(edge['start']))

    print(edge_dict)
    print('=='*10)
    # 准备所有网络层的参数、起始点的id、中间节点的名称和特殊节点的id
    param_dict = {}
    start = ''
    end = ''
    layers_dict = {}  # 网络名、配置id、输出id
    super_layer = ['Add']
    super_ids = []
    super_ids_dict = {}
    for node in nodes:
        if node['title'] == 'Params':
            param_dict[edge_dict[str(node['outputs'][0]['id'])][0]] = node['content']['value']
        elif node['title'] == 'Input':
            start = str(node['outputs'][0]['id'])
        elif node['title'] == 'Output':
            end = str(node['inputs'][0]['id'])
        else:
            layers_dict[str(node['inputs'][0]['id'])] = [node['title'], str(node['inputs'][1]['id']),
                                                         str(node['outputs'][0]['id'])]

        if node['title'] in super_layer:
            super_ids.append(str(node['inputs'][0]['id']))
            super_ids.append(str(node['inputs'][1]['id']))
            super_ids_dict[str(node['inputs'][0]['id'])] = str(node['outputs'][0]['id'])
            super_ids_dict[str(node['inputs'][1]['id'])] = str(node['outputs'][0]['id'])
    print(layers_dict)
    print('==' * 10)
    print(param_dict)
    print('==' * 10)
    # 实例化从起点到终点的所有层
    Node_Layers = []
    while start != end:
        # print(edge_dict[start])
        # print(layers_dict[edge_dict[start][1]][2])
        # print(param_dict[layers_dict[edge_dict[start][1]][1]])
        # ss
        # print(len(start))
        # ss
        # ss
        if len(start) != 2:
            print(start)
            if isinstance(start, list):
                start = start[0]
            if len(edge_dict[start]) == 1:
                Node_Layers.append(eval(
                    f'nn.{layers_dict[edge_dict[start][0]][0]}(**{param_dict[layers_dict[edge_dict[start][0]][1]]})'))
                start = layers_dict[edge_dict[start][0]][2]
                # print(edge_dict[start], end)
                if edge_dict[start][0] == end:
                    break
                # ss
                # pass
            else:
                Node_Layers.append(
                    [eval(
                        f'nn.{layers_dict[edge_dict[start][0]][0]}(**{param_dict[layers_dict[edge_dict[start][0]][1]]})'),
                        eval(
                            f'nn.{layers_dict[edge_dict[start][1]][0]}(**{param_dict[layers_dict[edge_dict[start][1]][1]]})')])
                start = [layers_dict[edge_dict[start][0]][2], layers_dict[edge_dict[start][1]][2]]
        else:
            # print(edge_dict[start[0]][0])
            # print(edge_dict[start[1]][0])
            # print(super_ids)
            # ss
            if edge_dict[start[0]][0] in super_ids:
                if edge_dict[start[1]][0] in super_ids:
                    # print(edge_dict[super_ids_dict[edge_dict[start[1]][0]]])
                    # print(end)
                    # ss
                    if layers_dict[edge_dict[start[0]][0]][0] == 'Add':
                        Node_Layers.append(lambda x, y: x + y)
                    # layers_dict[edge_dict[start][0]][2]
                    start = super_ids_dict[edge_dict[start[1]][0]]
                    # start = edge_dict[super_ids_dict[edge_dict[start[1]][0]]][0]
                    # print(start)
                    # print(end)
                    # ss
                else:
                    if isinstance(Node_Layers[-1][1], list):
                        Node_Layers[-1][1].append(eval(
                            f'nn.{layers_dict[edge_dict[start[1]][0]][0]}(**{param_dict[layers_dict[edge_dict[start[1]][0]][1]]})'))
                        start[1] = layers_dict[edge_dict[start[1]][0]][2]
                    else:
                        Node_Layers[-1][1] = [Node_Layers[-1][1]]
                        start[1] = layers_dict[edge_dict[start[1]][0]][2]
            else:
                if isinstance(Node_Layers[-1][0], list):
                    Node_Layers[-1][0].append(eval(
                        f'nn.{layers_dict[edge_dict[start[0]][0]][0]}(**{param_dict[layers_dict[edge_dict[start[0]][0]][1]]})'))
                    start[0] = layers_dict[edge_dict[start[0]][0]][2]
                else:
                    Node_Layers[-1][0] = [Node_Layers[-1][0]]
                    start[0] = layers_dict[edge_dict[start[0]][0]][2]
    # print()
        print(Node_Layers)
    return Node_Layers

def flatten_list(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result
# nn_layers = ['Conv2d', 'Linear', 'ReLU', 'Dropout', 'BatchNorm2d', 'MaxPool2d']
class NodeNet(nn.Module):
    def __init__(self, Node_Layers):
        super(NodeNet, self).__init__()
        self.Node_Layers = Node_Layers
        self._move_modules_to_device()
        # self.Node_Layers1 = nn.ModuleList(flatten_list(Node_Layers))

    def forward(self, din):
        return compose(din, self.Node_Layers)

    def parameters(self, recurse=True):
        # 收集所有参数
        params = []
        for layer in flatten_list(self.Node_Layers):
            if isinstance(layer, nn.Module):
                params += list(layer.parameters(recurse=recurse))
        return iter(params)

    def _move_modules_to_device(self):
        for layer in flatten_list(self.Node_Layers):
            if isinstance(layer, nn.Module):
                layer.to(self.device)

    @property
    def device(self):
        return 'cuda'

# nn.Sequential
if __name__ == '__main__':
    # print(isinstance(nn.Linear(2,3),nn.Module))
    # ss
    with open('demo.json') as f:
        data = json.load(f)
    Node_Layers = nodes2net(data)
    print(Node_Layers)
    net = NodeNet(Node_Layers).cuda()
    # net.parameters()
    din = torch.ones(2, 3, 224, 224).cuda()
    # print(compose(1, [lambda x:x+1 for i in range(3)]))
    print(net(din).shape)
# # 保存模型结构和参数
# torch.save(net, 'nodenet.pth')
# # 加载模型结构和参数
# net = torch.load('nodenet.pth')
# nn.BatchNorm2d
# print((1,)+(2,))

# 实例化测试
# title = 'Linear'
# pas = {'in_features':10, 'out_features':10}
# net = eval(f'nn.{title}(**{pas})')
# print(net)


# 剩余未完成
# 1.多分支处理--最多两分支
# 2.递归运算--OK
