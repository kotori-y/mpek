from torch import nn


# class MLP(nn.Module):
#     def __init__(
#         self,
#         in_size,
#         output_sizes,
#         use_layer_norm=False,
#         activation=nn.ReLU,
#         dropout=0.0,
#         layernorm_before=False,
#         use_bn=False,
#     ):
#         super().__init__()
#         module_list = []
#         if not use_bn:
#             if layernorm_before:
#                 module_list.append(nn.LayerNorm(in_size))
#             for size in output_sizes:
#                 module_list.append(nn.Linear(in_size, size))
#                 if size != 1:
#                     module_list.append(activation())
#                     if dropout > 0:
#                         module_list.append(nn.Dropout(dropout))
#                 in_size = size
#             if not layernorm_before and use_layer_norm:
#                 module_list.append(nn.LayerNorm(in_size))
#         else:
#             for size in output_sizes:
#                 module_list.append(nn.Linear(in_size, size))
#                 if size != 1:
#                     module_list.append(nn.BatchNorm1d(size))
#                     module_list.append(activation())
#                 in_size = size
#
#         self.module_list = nn.ModuleList(module_list)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for item in self.module_list:
#             if hasattr(item, "reset_parameters"):
#                 item.reset_parameters()
#
#     def forward(self, x):
#         for item in self.module_list:
#             x = item(x)
#         return x


class MLP(nn.Module):
    """
    MLP
    """
    def __init__(self, layer_num, in_size, hidden_size, out_size, dropout_rate):
        super(MLP, self).__init__()

        layers = []

        if layer_num == 1:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU())

        else:
            for layer_id in range(layer_num):
                if layer_id == 0:
                    layers.append(nn.Linear(in_size, hidden_size))
                    layers.append(nn.Dropout(dropout_rate))
                    layers.append(nn.ReLU())
                elif layer_id < layer_num - 1:
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.Dropout(dropout_rate))
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Linear(hidden_size, out_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, dim).
        """
        return self.mlp(x)


class MLPwoLastAct(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))

            if dropout > 0:
                module_list.append(nn.Dropout(dropout))
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(activation())
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x