def parallel(x, funcs):
    if not isinstance(funcs, list) and not isinstance(funcs, tuple):
        return funcs(x)
    else:
        return compose(x, funcs)


def compose(x, funcs):
    # print(type(funcs), 11111111)
    if len(funcs) == 0:
        return x
    if len(funcs) == 1:
        return funcs[0](x)
    else:
        fun1 = funcs[0]
        fun2 = funcs[1:]
        # print(fun1, fun2)
        if not isinstance(fun1, list):
            return compose(fun1(x), fun2)
        elif isinstance(fun1, list):
            return compose(fun2[0](*[parallel(x, fun) for fun in fun1]), fun2[1:])


if __name__ == '__main__':
    a = lambda x: x + 1
    b = lambda x, y: x * y
    # 最开始的问题建模
    '''
    (a,[b,c],d,e,f,g,h)
    (a,b,[c,(d,e,f)],g,h)
    (a,[b,(c,[d,e],f)],g,h)
    ()串行计算、[]并行计算
    []里面一定只有两个元素
    现在假设函数就两种
    lambda x:x+1
    lambda x,y:x*y
    []后面的第一个函数属于第二种
    其余的都属于第一种
    请实现一个通用的满足上述要求的函数
    示例:输入1和(a,[b,c],d)返回9
    输入1和([a,(b,c)],d)返回6
    '''
    print(compose(1, ()))  # 1
    print(compose(1, (a,)))  # 2
    print(compose(1, (a, a, a, a)))  # 5
    print(compose(1, ([a, a], b, a, a)))  # 6
    print(compose(1, ([a, a], b)))  # 4
    print(compose(1, ([([a, a], b), a], b, a)))  # 9
    print(compose(1, ([a, (a, a)], b, a)))  # 7
    print(compose(1, ([a, (a, [a, a], b)], b, a)))  # 19
    print(compose(1, ([a, (a, a)], b, a, [a, (a, a)], b, a)))  # 73

    # 尝试：最外面的()替换为[]，想着能不能和昨天对接，结果可以
    print(compose(1, []))  # 1
    print(compose(1, [a]))  # 2
    print(compose(1, [a, a, a]))  # 4
    print(compose(1, [[a, a], b, a, a]))  # 6
    print(compose(1, [[a, a], b]))  # 4
    print(compose(1, [[([a, a], b), a], b, a]))  # 9
    print(compose(1, [[a, (a, a)], b, a]))  # 7
    print(compose(1, [[a, (a, [a, a], b)], b, a]))  # 19
    print(compose(1, [[a, (a, a)], b, a, [a, (a, a)], b, a]))  # 73

    # 意外收获：()全部替换为[]，也ok
    print(compose(1, [[[[a, a], b], a], b, a]))  # 9
    print(compose(1, [[a, [a, a]], b, a]))  # 7
    print(compose(1, [[a, [a, [a, a], b]], b, a]))  # 19
    print(compose(1, [[a, [a, a]], b, a, [a, [a, a]], b, a]))  # 73
