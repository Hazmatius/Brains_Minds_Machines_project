import torch
import torch.nn as nn
import numpy as np
import itertools


'''
This is a library of functions meant to make it easy to create trainable neural networks from what is basically a signal
flow diagram. The basic ontology of this library is as follows:

state_dict: a dictionary of named variables. All networks use this as a common resource, so interactions between
networks are through joint modification and use of variables in the state_dict.

network: a neural network. You specify the input and output variables. There are special factory functions that will
generate the code necessary to handle merging of inputs and forking of outputs.

adapter: an optimizer that takes as inputs the loss function to be minimized, the variables to compute that loss
function on, and the networks that should be adapted to minimize this loss function, as well as any necessary parameters
of the optimizer.

'''


def state_dict_from_spec(state_spec, kind):
    '''
    This function constructs a dictionary of state variables from a description (also encoded as a dictionary) of the
    final form of the state variable dictionary.
    :param state_spec: a dictionary with keys as variable names and values as tensor sizes
    :param kind: string saying whether to initialize tensors with zero or random values. Must be 'zeros' or 'rand'
    :return: returns an initialized state_dict
    '''
    terms = list(state_spec.keys())
    if kind == 'zeros':
        return {term: torch.zeros(1, *state_spec[term]) for term in terms}
    elif kind == 'rand':
        return {term: torch.rand(1, *state_spec[term]) for term in terms}
    else:
        raise Exception('kind must be one of "zeros" or "rand"')


# I guess I only really have this function because I hate having to remember how to concatenate tensors along features
# dimensions
def concat(tensors):
    '''
    :param tensors: the tensors to concatenate
    :return: the concatenated tensors
    '''
    return torch.cat(tensors, dim=1)


# def split_evenly(tensor, n):
#     assert torch.numel(tensor) % n == 0
#     chunk_size = int(torch.numel(tensor) / n)
#     return torch.split(tensor, chunk_size, dim=1)


def split(tensor, ks):
    '''

    :param tensor: the tensor to be split
    :param ks: a list of integers, each is the size of the output tensor
    :return: the list of split tensors
    '''
    # assert sum(ks) == torch.numel(tensor)
    return torch.split(tensor, ks, dim=1)


# we may have to do some fancy tensor copying for all of this to work out
def get_inflow_func(kind, terms_dict):
    '''
    Networks officially only take single-tensor inputs. However, we can simulate multiple inputs by simply concatenating
    the multiple tensors into a single tensor. In order to increase the speed and modularity of this operation, this
    function generates from a specifiction of the variables to concatenate, a function that automatically knows which
    variables in the input dictionary to look for.
    :param kind:
    :param terms_dict:
    :return:
    '''
    terms = list(terms_dict.keys())
    if kind == 'unity':
        assert len(terms) == 1
        def inflow_func(input_dict):
            return input_dict[terms[0]]
    elif kind == 'merge':
        def inflow_func(input_dict):
            return concat([input_dict[term] for term in terms])
    else:
        raise Exception('inflow_func kind must be one of either "unity" or "merge"')
    return inflow_func


def get_time_shift_func(vars, timeline):
    time_spec = []
    for i in range(len(timeline)-1):
        time_spec.append((timeline[i], timeline[i+1]))
    shift_map = list()
    for var in vars:
        for shift in time_spec:
            shift_map.append(('{}_{}'.format(var, shift[0]), '{}_{}'.format(var, shift[1])))
    def time_shift(state_dict):
        for shift in shift_map:
            state_dict[shift[0]] = state_dict[shift[1]]

    return time_shift


def get_outflow_func(kind, terms_dict):
    terms = list(terms_dict.keys())
    term_dims = [terms_dict[term] for term in terms]
    if kind == 'unity':
        assert len(terms) == 1
        def outflow_func(output_tensor, output_dict):
            output_dict[terms[0]] = output_tensor
    elif kind == 'fork':
        def outflow_func(output_tensor, output_dict):
            tensors = split(output_tensor, term_dims)
            for i in range(len(terms)):
                output_dict[terms[i]] = tensors[i]
    else:
        raise Exception('outflow_func kind must be one of either "unity" or "fork"')
    return outflow_func


def flat_unique(pylist):
    return list(set(itertools.chain.from_iterable(pylist)))


def order_funcs(state_spec, func_specs):
    '''
    This function takes a state_spec dictionary (keys are variable names, values are tensor sizes) and a func_specs
    dictionary (keys are function names, values are function specifications which include input and output tensors).
    It then generates a computation graph and orders it so that functions are executed in a way that they always have
    pre-computed inputs. The function returns an order of function execution.
    :param state_spec:
    :param func_specs:
    :return:
    '''
    nodes = dict()
    for key, value in state_spec.items():
        nodes[key] = Node(key)
    edges = list()
    for func_spec in func_specs:
        in_nodes = [nodes[node_name] for node_name in func_spec['inputs']]
        out_nodes = [nodes[node_name] for node_name in func_spec['outputs']]
        edge_name = func_spec['func_name']
        edges.append(Edge(edge_name, in_nodes, out_nodes))
    loops = False
    err_msg = ''
    for node_name, node in nodes.items():
        ancestors = node.get_ancestors(set())
        if node in ancestors:
            err_msg += '{} depends on itself\n'.format(node_name)
            loops = True
    if loops:
        raise Exception('Error, computation graph contains loops:\n{}'.format(err_msg))
    input_nodes = list()
    for node_name, node in nodes.items():
        if not node.in_edges:
            input_nodes.append(node)
    execution_list = list()
    while edges:
        for edge in edges:
            if set(edge.in_nodes).issubset(set(input_nodes)):
                execution_list.append(edge)
                input_nodes += edge.out_nodes
        edges = list(set(edges).difference(set(execution_list)))
    execution_list = [edge.name for edge in execution_list]
    return execution_list


class Edge(object):
    def __init__(self, name, in_nodes, out_nodes):
        self.name = name
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

        for node in self.in_nodes:
            node.add_out_edge(self)

        for node in self.out_nodes:
            node.add_in_edge(self)


class Node(object):
    def __init__(self, name):
        self.name = name
        self.in_edges = list()
        self.out_edges = list()

    def add_in_edge(self, edge):
        self.in_edges.append(edge)

    def add_out_edge(self, edge):
        self.out_edges.append(edge)

    def get_parent_nodes(self):
        return flat_unique([edge.in_nodes for edge in self.in_edges])

    def get_ancestors(self, used_nodes):
        used_nodes.add(self)
        ancestors = set()
        parent_nodes = self.get_parent_nodes()
        for parent in parent_nodes:
            if parent not in used_nodes:
                ancestors = ancestors.union(parent.get_ancestors(used_nodes))
            else:
                ancestors.add(parent)
        return ancestors


class Network(object):
    def __init__(self, state_spec, state_dict):
        self.state_spec = state_spec
        self.state_dict = state_dict
        self.func_specs = list()
        self.functions = dict()
        self.adapters = list()
        self.exec_order = None

    def compile(self):
        self.exec_order = order_funcs(self.state_spec, self.func_specs)

    def add_function(self, func_spec):
        if func_spec['type'] == 'MLP':
            func = MLP(**func_spec)
        elif func_spec['type'] == 'Add':
            func = Add(**func_spec)
        elif func_spec['type'] == 'Mult':
            func = Mult(**func_spec)
        elif func_spec['type'] == 'Nonlin':
            func = Nonlin(**func_spec)
        else:
            raise Exception('Unknown function type "{}"'.format(func_spec['type']))
        self.functions[func_spec['func_name']] = func
        self.func_specs.append(func_spec)

    def add_functions(self, func_specs):
        for func_spec in func_specs:
            self.add_function(func_spec)

    def add_adapter(self, adapter_spec):
        self.adapters.append(Adapter(**adapter_spec))

    def add_adapters(self, adapter_specs):
        for adapter_spec in adapter_specs:
            adapter_spec['models'] = [self.functions[model_name] for model_name in adapter_spec['models']]
            self.add_adapter(adapter_spec)

    def forward(self):
        for func_name in self.exec_order:
            self.functions[func_name].forward(self.state_dict)

    def adapt(self):
        for adapter in self.adapters:
            adapter.adapt(self.state_dict)


class Adapter(object):
    def __init__(self, **kwargs):
        self.target = kwargs['target']
        self.output = kwargs['output']
        parameters = itertools.chain.from_iterable([model.parameters() for model in kwargs['models']])
        self.optimizer = kwargs['optimizer'](parameters, **kwargs['optimizer_args'])
        self.loss_module = kwargs['loss_module']

    def adapt(self, state_dict):
        loss = self.loss_module(state_dict[self.output], state_dict[self.target])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Layer(nn.Module):
    def __init__(self, **kwargs):
        super(Layer, self).__init__()
        self.affine = nn.Linear(kwargs['in_features'], kwargs['out_features'])
        self.nonlin = kwargs['nonlin']()

    def forward(self, input):
        return self.nonlin(self.affine(input))


class Function(nn.Module):
    def __init__(self, **kwargs):
        super(Function, self).__init__()
        self.func_name = kwargs['func_name']
        self.inputs = kwargs['inputs']
        self.outputs = kwargs['outputs']
        self.state_spec = kwargs['state_spec']
        self.input_specs = {term: self.state_spec[term] for term in self.inputs}
        self.output_specs = {term: self.state_spec[term] for term in self.outputs}

    def forward(self, state_dict):
        pass


class SMult(Function):
    def __init__(self, **kwargs):
        super(SMult, self).__init__(**kwargs)
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.input_specs[self.inputs[0]] == self.output_specs[self.outputs[0]]
        self.scalar = kwargs['scalar']

    def forward(self, state_dict):
        state_dict[self.outputs[0]] = self.scalar * state_dict[self.inputs[0]]


class Add(Function):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)
        assert len(self.inputs) == 2
        assert len(self.outputs) == 1
        in1size = self.input_specs[self.inputs[0]]
        in2size = self.input_specs[self.inputs[1]]
        outsize = self.output_specs[self.outputs[0]]
        assert in1size == in2size and in2size == outsize

    def forward(self, state_dict):
        state_dict[self.outputs[0]] = self.state_dict[self.inputs[0]] + self.state_dict[self.inputs[1]]


class Mult(Function):
    def __init__(self, **kwargs):
        super(Mult, self).__init__(**kwargs)
        assert len(self.inputs) == 2
        assert len(self.outputs) == 1
        in1size = self.input_specs[self.inputs[0]]
        in2size = self.input_specs[self.inputs[1]]
        outsize = self.output_specs[self.outputs[0]]
        assert in1size == in2size and in2size == outsize

    def forward(self, state_dict):
        state_dict[self.outputs[0]] = state_dict[self.inputs[0]] * state_dict[self.inputs[1]]


class Nonlin(Function):
    def __init__(self, **kwargs):
        super(Nonlin, self).__init__(**kwargs)
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        insize = self.input_specs[self.inputs[0]]
        outsize = self.output_specs[self.outputs[0]]
        assert insize == outsize
        self.nonlin = kwargs['nonlin']()

    def forward(self, state_dict):

        state_dict[self.outputs[0]] = self.nonlin(state_dict[self.inputs[0]])


class MLP(Function):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.in_dim = sum([self.state_spec[inpt][0] for inpt in self.inputs])
        self.out_dim = sum([self.state_spec[outp][0] for outp in self.outputs])

        layer_spec = kwargs['layer_spec']
        if 'n_layers' in layer_spec:
            self.layer_dims = [int(round(i)) for i in np.linspace(self.in_dim, self.out_dim, layer_spec['n_layers']+1)]

        # this will always be unity (take the specified term and return the tensor) or
        # merge (take the specified terms and concatenate them, returning a tensor)
        if len(self.inputs) == 1:
            self.inflow_func = get_inflow_func('unity', self.input_specs)
        elif len(self.inputs) > 1:
            self.inflow_func = get_inflow_func('merge', self.input_specs)
        else:
            raise Exception('Number of input terms must be greater than 0')
        if len(self.outputs) == 1:
            self.outflow_func = get_outflow_func('unity', self.output_specs)
        elif len(self.outputs) > 1:
            self.outflow_func = get_outflow_func('fork', self.output_specs)
        else:
            raise Exception('Number of output terms must be greater than 0')

        layers = list()
        for i in range(1, len(self.layer_dims)):
            layer_args = {
                'in_features': self.layer_dims[i-1],
                'out_features': self.layer_dims[i],
                'nonlin': kwargs['nonlin']
            }
            layers.append(Layer(**layer_args))
        self.layers = nn.ModuleList(layers)

    def forward(self, state_dict):
        # print('running...')
        x = self.inflow_func(state_dict)
        # print('finished running.')
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
            # print(x)
        self.outflow_func(x, state_dict)

