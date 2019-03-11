# -*- coding: utf-8 -*-

class NetOutput:
    def __init__(self):
        self.sum_inputs   = [] # <number>, sum_inputs through time. (每一个时间步的输入值)
        self.output_values = [] # <number>, Output values through time.(每一个时间步的输出值， output_values 的上一刻object 就是 b[t-1][h])
        self._initialize_outputs()

    def add_sum_input(self, singal=0.0):
        self.sum_inputs.append(singal)

    def add_output_value(self, output=0.0):
        self.output_values.append(output)

    def refresh(self):
        del self.sum_inputs[:]
        del self.output_values[:]
        self._initialize_outputs()

    def remove_last_output(self):
        self.sum_inputs.pop()
        self.output_values.pop()

    # 对 t=0 时刻进行 0 值填充
    def _initialize_outputs(self):
        self.sum_inputs.append(0.0)
        self.output_values.append(0.0)

    @property
    # 当前刻的输出或为在 Forward 运算时的上一刻的输入值
    def last_sum_input(self):
        return self.sum_inputs[-1] # last object

    @property
    # 当前刻的输出或为在 Forward 运算时的上一刻输出值， e.g. b[t][h] or b[t-1][h]
    def last_output_value(self):
        return self.output_values[-1]

    @property
    # 取出上一刻的输出值， e.g. b[t-1][h]
    def previous_output(self):
        index = self.previous_index
        return self.output_values[index] if index >= 0 else 0.0

    @property
    # 取出上一刻的输入值， e.g. a[t-1][h]
    def previous_sum_input(self):
        index = self.previous_index
        return self.sum_inputs[index] if index >= 0 else 0.0

    @property
    # Last moment index for outputValues and sumInputs.
    def previous_index(self):
        # 当前 Index 的上一刻 Index, 故相对应的索引位置须 -2 （注： -1 为当前的 Index Number）
        return self.outputs_count - 2

    @property
    # Same as time length.
    def outputs_count(self):
        return len(self.output_values)