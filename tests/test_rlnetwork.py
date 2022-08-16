from msilib import OpenDatabase
import unittest
from tianshou_agents.networks import RLNetwork
from numbers import Number
from torch import nn
import torch

def make_net(input_shape, output_shape, device):
    return nn.Sequential(
        nn.Linear(input_shape, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, output_shape),
    ).to(device)

class TestNetSimple(nn.Module):
    def __init__(self, input_shape, output_shape, device):
        super().__init__()

        self.input_dense = nn.Linear(input_shape, 256)
        self.input_acti = nn.ReLU()

        self.dense1 = nn.Linear(256, 256)
        self.activation1 = nn.ReLU()

        self.output_dense = nn.Linear(256, output_shape)

    def forward(self, obs, state=None):
        y = self.input_dense(obs)
        y = self.input_acti(y)

        y = self.dense1(y)
        y = self.activation1(y)

        y = self.output_dense(y)

        return y

class TestNet(nn.Module):
    def __init__(self, input_shape, output_shape, device):
        super().__init__()

        if isinstance(input_shape, Number):
            input_shape = [input_shape]

        self.input_dense = [nn.Linear(ninp, 256) for ninp in input_shape]
        self.input_activations = [nn.ReLU() for _ in input_shape]

        self.dense1 = nn.Linear(256, 256)
        self.activation1 = nn.ReLU()

        self.output_dense = nn.Linear(256, output_shape)

    def forward(self, obs, state=None):
        y = 0

        for x, dense, acti in zip(obs, self.input_dense, self.input_activations):
            y = y + acti(dense(x))
        
        y = self.dense1(y)
        y = self.activation1(y)

        y = self.output_dense(y)

        return y

class TestNetAct(nn.Module):
    def __init__(self, input_shape, output_shape, device):
        super().__init__()
        obs_shape, act_shape = input_shape

        if isinstance(obs_shape, Number):
            obs_shape = [obs_shape]

        if isinstance(act_shape, Number):
            act_shape = [act_shape]

        self.obs_dense = [nn.Linear(ninp, 256) for ninp in obs_shape]
        self.obs_activations = [nn.ReLU() for _ in obs_shape]

        self.act_dense = [nn.Linear(ninp, 256) for ninp in act_shape]
        self.act_activations = [nn.ReLU() for _ in act_shape]

        self.dense1 = nn.Linear(256, 256)
        self.activation1 = nn.ReLU()

        self.output_dense = nn.Linear(256, output_shape)

    def forward(self, obs, act, state=None):
        y = 0

        for x, dense, acti in zip(obs, self.obs_dense, self.obs_activations):
            y = y + acti(dense(x))

        for x, dense, acti in zip(act, self.act_dense, self.act_activations):
            y = y + acti(dense(x))
            
        y = self.dense1(y)
        y = self.activation1(y)

        y = self.output_dense(y)

        return y

class TestConvNet(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(3, 10, (3, 3))
        self.dense1 = nn.Linear(640, 25)
        self.dense2 = nn.Linear(25, output_shape)

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
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(3, 10, (3, 3))
        self.dense1 = nn.Linear(640, 25)
        self.dense2 = nn.Linear(9, 25)
        self.dense3 = nn.Linear(25, output_shape)

    def forward(self, inputs, state=None):
        x1, x2 = inputs
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
            state_shape, action_shape, model=TestNetSimple,
            device=device, flatten=False
        )

        dummy_inputs = torch.ones(5, state_shape)
        output = qnet(dummy_inputs)

        self.assertEqual(output[0].shape, (5, 7))

    def testTwoInputs(self):
        state_shape = ((10,), (9,))

        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape, model=TestNet,
            device=device, flatten=False
        )

        dummy_batch = [
            (torch.ones(10), torch.ones(9)) for _ in range(5)
        ]
        
        output = qnet(dummy_batch)
        self.assertEqual(output[0].shape, (5, 7))

    def testTwoInputsActionsAsInput(self):
        state_shape = ((10,), (9,))
        action_shape = 7
        device = 'cpu'

        qnet = RLNetwork(
            state_shape, action_shape, model=TestNetAct,
            device=device, flatten=False, actions_as_input=True
        )

        dummy_batch = [
            (torch.ones(10), torch.ones(9)) for _ in range(5)
        ]

        dummy_actions = torch.ones(5, action_shape)
        output = qnet(dummy_batch, dummy_actions)

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

        dummy_batch = [
            [torch.ones(3, 10, 10), torch.ones(9)] for _ in range(5)
        ]
        
        output = qnet(dummy_batch)
        self.assertEqual(output[0].shape, (5, 7))

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
