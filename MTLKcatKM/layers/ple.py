import torch
import torch.nn as nn
import torch.nn.functional as F

from MTLKcatKM.layers import MLP


class Expert(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, expert_hid_layer=3, dropout_rate=0.3):
        super(Expert, self).__init__()
        self.expert = MLP(
            in_size=in_size,
            hidden_size=hidden_size,
            out_size=out_size,
            layer_num=expert_hid_layer,
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        return self.expert(x)


class Gate(nn.Module):
    def __init__(self, input_size, output_size):
        super(Gate, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc1(x)


class ExpertModule(nn.Module):
    def __init__(self, experts_in, experts_out, experts_hidden, expert_hid_layer=3, dropout_rate=0.3, num_experts=5,
                 num_tasks=2):
        super(ExpertModule, self).__init__()

        self.num_experts = num_experts
        self.experts_out = experts_out

        self.share_expert = nn.ModuleList(
            [
                Expert(
                    in_size=experts_in,
                    out_size=experts_out,
                    hidden_size=experts_hidden,
                    expert_hid_layer=expert_hid_layer,
                    dropout_rate=dropout_rate
                ) for _ in range(num_experts)
            ]
        )

        self.task_experts = nn.ModuleList([])
        for _ in range(num_tasks):
            task_expert = nn.ModuleList(
                [
                    Expert(
                        in_size=experts_in,
                        out_size=experts_out,
                        hidden_size=experts_hidden,
                        expert_hid_layer=expert_hid_layer,
                        dropout_rate=dropout_rate
                    ) for _ in range(num_experts)
                ]
            )
            self.task_experts.append(task_expert)

        # self.expert_activation = nn.ReLU()

    def forward(self, share_x, task_x):
        assert len(task_x) == len(self.task_experts)

        share_expert_out = [e(share_x) for e in self.share_expert]
        share_expert_out = torch.concat(share_expert_out, dim=0).view(-1, self.num_experts, self.experts_out)
        # share_expert_out = self.expert_activation(share_expert_out)

        task_expert_out_list = []
        for i, task_expert in enumerate(self.task_experts):
            task_expert_out = [e(task_x[i]) for e in task_expert]
            task_expert_out = torch.concat(task_expert_out, dim=0).view(-1, self.num_experts, self.experts_out)
            # task_expert_out = self.expert_activation(task_expert_out)
            task_expert_out_list.append(task_expert_out)

        return share_expert_out, task_expert_out_list


class GateModule(nn.Module):
    def __init__(self, gate_in, num_experts, num_tasks):
        super(GateModule, self).__init__()

        self.share_gate = Gate(gate_in, num_experts * (num_tasks + 1))
        self.task_gates = nn.ModuleList(
            [Gate(gate_in, num_experts * 2) for _ in range(num_tasks)]
        )
        self.gate_activation = nn.Softmax(dim=-1)

    def forward(self, share_x, task_x):
        assert len(task_x) == len(self.task_gates)

        share_gate_out = self.share_gate(share_x)
        share_gate_out = self.gate_activation(share_gate_out)

        task_gate_out_list = [e(task_x[i]) for i, e in enumerate(self.task_gates)]

        return share_gate_out, task_gate_out_list


class PleLayer(nn.Module):

    def __init__(
            self,
            experts_in, experts_out, experts_hidden,
            expert_hid_layer=3, dropout_rate=0.3,
            num_experts=5, num_tasks=2
    ):
        super(PleLayer, self).__init__()

        self.experts = ExpertModule(
            experts_in, experts_out, experts_hidden, expert_hid_layer, dropout_rate, num_experts, num_tasks
        )

        self.gates = GateModule(experts_in, num_experts, num_tasks)

    def forward(self, share_x, task_x):
        share_expert_out, task_expert_out_list = self.experts(share_x, task_x)
        share_gate_out, task_gate_out_list = self.gates(share_x, task_x)

        task_out_list = []
        for i in range(len(task_x)):
            task_expert_out = task_expert_out_list[i]
            task_gate_out = task_gate_out_list[i]

            task_out = torch.cat([share_expert_out, task_expert_out], dim=1)
            task_out = torch.einsum('be,beu -> beu', task_gate_out, task_out)
            task_out = task_out.sum(dim=1)

            task_out_list.append(task_out)

        share_out = torch.cat([share_expert_out, *task_expert_out_list], dim=1)
        share_out = torch.einsum('be,beu -> beu', share_gate_out, share_out)
        share_out = share_out.sum(dim=1)

        return share_out, task_out_list


class PLE(nn.Module):
    def __init__(self, experts_in, experts_out,
                 experts_hidden, expert_hid_layer=3,
                 dropout_rate=0.3, num_experts=5, num_tasks=2, num_ple_layers=1):
        super(PLE, self).__init__()

        self.layers = nn.ModuleList([])
        self.num_tasks = num_tasks

        for i in range(num_ple_layers):
            if i == 0:
                layer = PleLayer(
                    experts_in, experts_out, experts_hidden,
                    expert_hid_layer, dropout_rate,
                    num_experts, num_tasks
                )
            else:
                layer = PleLayer(
                    experts_out, experts_out, experts_hidden,
                    expert_hid_layer, dropout_rate,
                    num_experts, num_tasks
                )

            self.layers.append(layer)

    def forward(self, x):
        share_x, task_x = x, [x for _ in range(self.num_tasks)]

        for layer in self.layers:
            share_x, task_x = layer(share_x, task_x)

        return task_x


if __name__ == "__main__":
    EXPERTS_IN = 900
    EXPERTS_OUT = 256
    EXPERTS_HIDDEN = 256
    EXPERT_HID_LAYER = 1
    DROPOUT_RATE = 0.3
    NUM_EXPERTS = 5
    NUM_TASKS = 1

    BS = 1

    x = torch.randn(BS, EXPERTS_IN)

    model = PLE(EXPERTS_IN, EXPERTS_OUT, EXPERTS_HIDDEN, EXPERT_HID_LAYER, DROPOUT_RATE, NUM_EXPERTS, NUM_TASKS, num_ple_layers=2)
    model(x)

    # gate = GateModule(EXPERTS_IN, NUM_EXPERTS, NUM_TASKS)
    # gate(x, [x, x])

    # gate = GatingNetwork(INPUT_UNITS, UNITS, NUM_EXPERTS, SELECTORS)
    # gate(x, x, x)
