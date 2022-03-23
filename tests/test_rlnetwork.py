import unittest
from tianshou_agents.networks import RLNetwork
from numbers import Number
from torch import nn
import torch

def make_net(num_inputs, num_outputs, device):
    return nn.Sequential(
        nn.Linear(num_inputs, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_outputs),
    )
    
class TestNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        super().__init__()

        if isinstance(num_inputs, Number):
            num_inputs = [num_inputs]

        self.input_dense = [nn.Linear(ninp, 256) for ninp in num_inputs]
        self.input_activations = [nn.ReLU() for _ in num_inputs]

        self.dense1 = nn.Linear(256, 256)
        self.activation1 = nn.ReLU()

        self.output_dense = nn.Linear(256, num_outputs)

    def forward(self, *args, state=None):
        y = 0

        for x, dense, act in zip(args, self.input_dense, self.input_activations):
            y = y + act(dense(x))
        
        y = self.dense1(y)
        y = self.activation1(y)

        y = self.output_dense(y)

        return y

class TestConvNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, **kwargs):
        super().__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(3, 10, (3, 3))
        self.dense1 = nn.Linear(640, 25)
        self.dense2 = nn.Linear(25, num_outputs)

    def forward(self, x1, state=None):
        y = self.conv(x1)
        y = self.relu(y)
        y = self.flatten(y)
        y = self.dense1(y)
        y = self.relu(y)
        y = self.dense2(y)
        y = self.relu(y)

        return y

class TestConvNet2(nn.Module):
    def __init__(self, num_inputs, num_outputs, **kwargs):
        super().__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(3, 10, (3, 3))
        self.dense1 = nn.Linear(640, 25)
        self.dense2 = nn.Linear(9, 25)
        self.dense3 = nn.Linear(25, num_outputs)

    def forward(self, x1, x2, state=None):
        y1 = self.conv(x1)
        y1 = self.relu(y1)
        y1 = self.flatten(y1)
        y1 = self.dense1(y1)
        y1 = self.relu(y1)
        
        y2 = self.dense2(x2)
        y2 = self.relu(y2)

        y = self.dense3(y1 + y2)
        y = self.relu(y)

        return y

class TestRLNetwork(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFlat(self):
        state_shape = 10
        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape,
            model=make_net, device=device
        )

        dummy_input = torch.ones(5, state_shape)
        output = qnet(dummy_input)
        self.assertEqual(output[0].shape, (5, 7))

    def testActionsAsInput(self):
        state_shape = 10
        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape, model=make_net,
            actions_as_input=True, device=device
        )

        dummy_input = torch.ones(5, state_shape + action_shape)
        output = qnet(dummy_input)

        self.assertEqual(output[0].shape, (5, 1))

    def testComplex(self):
        state_shape = 10
        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape, model=TestNet,
            device=device, flatten=False
        )

        dummy_inputs = (torch.ones(5, state_shape),)
        output = qnet(*dummy_inputs)

        self.assertEqual(output[0].shape, (5, 7))

    def testTwoInputs(self):
        state_shape = [10, 9]
        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape, model=TestNet,
            device=device, flatten=False
        )

        dummy_inputs = (torch.ones(5, 10), torch.ones(5, 9))
        output = qnet(dummy_inputs)

        self.assertEqual(output[0].shape, (5, 7))

    def testTwoInputsActionsAsInput(self):
        state_shape = [10, 9]
        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape, model=TestNet,
            device=device, flatten=False, actions_as_input=True
        )

        dummy_inputs = (torch.ones(5, 10), torch.ones(5, 9))
        output = qnet(dummy_inputs)

        self.assertEqual(output[0].shape, (5, 1))

    def test2D(self):
        state_shape = (3, 10, 10)
        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape, model=TestConvNet,
            device=device, flatten=False
        )

        dummy_inputs = torch.ones(50, 3, 10, 10)
        output = qnet(dummy_inputs)

        self.assertEqual(output[0].shape, (50, 7))

    def testTwoInputs2D(self):
        state_shape = [(3, 10, 10), 9]
        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape, model=TestConvNet2,
            device=device, flatten=False
        )

        dummy_inputs = (torch.ones(50, 3, 10, 10), torch.ones(50, 9))
        output = qnet(dummy_inputs)

        self.assertEqual(output[0].shape, (50, 7))

    def testMLP(self):
        state_shape = 10
        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape,
            device=device,
            hidden_sizes=[128, 128]
        )

        dummy_input = torch.ones(5, state_shape)
        output = qnet(dummy_input)
        self.assertEqual(output[0].shape, (5, 7))

    def testDuelingMLP(self):
        state_shape = 10
        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape,
            device=device,
            hidden_sizes=[128, 128],
            dueling_param=(
                {"hidden_sizes": [128, 128]}, # Q_param
                {"hidden_sizes": [128, 128]} # V_param
            )
        )

        dummy_input = torch.ones(5, state_shape)
        output = qnet(dummy_input)
        self.assertEqual(output[0].shape, (5, 7))
