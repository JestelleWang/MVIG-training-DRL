# Deep Reinforcement Learning

Reinforcement learning explicitly considers the whole problem of a goal-directed agent interacting with an uncertain environment. It faces the challenge of the trade-off between exploration and exploitation. That is to say, *the agent has to exploit what it already knows in order to obtain reward, but it also has to explore in order to make better action selections in the future.* The agent must try a variety of actions and progressively favor those gains more rewards.

Beyond the agent and the environment, one can identify four main subelements of a reinforcement learning system:

- *Policy*: the learning agent's way of behaving at a given time
- *reward signal*: the goal in a reinforcement learning problem
- *value function*: specifies what is good in the long run
- *model*: something that mimics the behavior of the environment that allows inferences to be made about how the environment will behave.

## Use PyTorch to train a DQN

```
class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		self.head = nn.Linear(448, 2)
	
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1)
```

In fact, the *DQN* is just use CNN to transfer the image input to states of Q-Learning.

 