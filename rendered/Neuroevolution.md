# Neuroevolution

Neuroevolution uses evolutionary algorithms to optimise neural networks. Before we start with the implementation of neural networks, let's import required dependencies.


```python
import random
import math
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

from graphviz import Digraph
```

## Neural Networks

Neural networks consist of neurons and weighted connections between these neurons. Each neuron represents a processing unit in which an activation function is applied to the weighted sum of all incoming connections. After a neuron has been activated, its activation signal is further propagated into the network.


```python
# Basic neuron definition from which all neuron genes will be derived.
class Neuron:

    def __init__(self, uid: str):
        self.uid = uid
        self.signal_value = 0
        self.activation_value = 0
        self.incoming_connections = []
```

### Input Neuron

Input neurons receive a signal from the environment (input feature) and propagate it into the network. Since we are interested in the raw input signal, we refrain from applying any activation functions within the input layer.


```python
class InputNeuron(Neuron):

    def __init__(self, uid: str):
        super().__init__(uid)

    def activate(self) -> float:
        self.activation_value = self.signal_value
        return self.activation_value
```

### Hidden Neuron

Hidden neurons reside between input and output neurons. They increase the capacity of a neural network, in other words, we require more hidden neurons for complex tasks than for simple ones. The number and distribution of neurons is an optimisation task on its own and one of the main reasons to use neuroevolution. As an activation function, we will use the sigmoid function, which maps any input value to $[0,1]$.


```python
def sigmoid(x: float):
    return 1 / (1 + math.exp(-x)) 

x_range = np.arange(-10, 10, 0.01)
sigmoid_values = [sigmoid(x) for x in x_range]
plt.plot(x_range, sigmoid_values)
plt.title('Sigmoid function', fontsize=15)
plt.show()
```


    
![png](13/output_7_0.png)
    



```python
class HiddenNeuron(Neuron):

    def __init__(self, uid: str):
        super().__init__(uid)

    def activate(self) -> float:
        self.activation_value = sigmoid(self.signal_value)
        return self.activation_value
```

### Output Neuron

For our output neurons, we have to choose an appropriate activation function that matches the given task. Later, we will try to solve a binary decision problem in which a robot can move to the right or left. Hence, we could use a sigmoid function, whose output is treated as a probability for choosing one of the two classes. However, due to the neuroevolution algorithm we will implement, we add two output neurons to the output layer, assign one output neuron to each class, and choose the class whose output neuron has the highest activation value.


```python
class OutputNeuron(Neuron):

    def __init__(self, uid: str):
        super().__init__(uid)
        self.uid = uid

    def activate(self) -> float:
        self.activation_value = self.signal_value
        return self.activation_value
```

### Connections

Our neurons are rather useless as long as they are not connected. We define connections by their source and destination neurons. Furthermore, we assign a weight to each connection to specify the strength of a link between two neurons.


```python
class Connection:

    def __init__(self, source: Neuron, target: Neuron, weight: float):
        self.source = source
        self.target = target
        self.weight = weight
```

### Network Definition

With our neurons and connections defined, we can now assemble both components in layers to build a neural network.


```python
class Network:
    def __init__(self, layers: dict[float, list[Neuron]], connections: list[Connection]):
        self.layers = layers
        self.connections = connections
        self.generate()

    # Traverse list of connections and add incoming connections to neurons.
    def generate(self):
        for connection in self.connections:
            connection.target.incoming_connections.append(connection)
```

Let's build a network with five input neurons, three hidden neurons and two output neurons.


```python
def gen_simple_network() -> Network:
    input_neurons = [InputNeuron(f"I{x}") for x in range(5)]
    hidden_neurons = [HiddenNeuron(f"H{x}") for x in range(3)]
    output_neurons = [OutputNeuron(f"O{x}") for x in range(5)]

    # Assemble in layers with 0 representing the input layer and 1 the output layer.
    layers = {0: input_neurons, 0.5: hidden_neurons, 1: output_neurons}

    # Generate connections from every input neuron to each hidden neuron
    connections = []
    for input_neuron in input_neurons:
        for hidden_neuron in hidden_neurons:
            weight = random.uniform(-1, 1)
            connections.append(Connection(input_neuron, hidden_neuron, weight))

    # Generate connections from every hidden neuron to each output neuron
    for hidden_neuron in hidden_neurons:
        for output_neuron in output_neurons:
            weight = random.uniform(-1, 1)
            connections.append(Connection(hidden_neuron, output_neuron, weight))

    return Network(layers, connections)


net = gen_simple_network()
```

We can visualise our generated network using the Graphviz library.


```python
class Network(Network):
    def show(self) -> Digraph:
        dot = Digraph(graph_attr={'rankdir': 'BT', 'splines': "line"})
        # Use sub graphs to position input neurons are at the bottom and output neurons at the top.
        input_graph = Digraph(graph_attr={'rank': 'min', 'splines': "line"})
        output_graph = Digraph(graph_attr={'rank': 'max', 'splines': "line"})
        hidden_graph = Digraph(graph_attr={'splines': "line"})

        # Traverse network from input to output layer and assign neurons to corresponding subgraph.
        layer_keys = list(self.layers.keys())
        layer_keys.sort()
        for layer in layer_keys:
            for neuron in self.layers.get(layer):
                if layer == 0:
                    input_graph.node(neuron.uid, color='black', fillcolor='white', style='filled')
                elif layer == 1:
                    output_graph.node(neuron.uid, color='black', fillcolor='white', style='filled')
                else:
                    hidden_graph.node(neuron.uid, color='black', fillcolor='white', style='filled')

        # Combine the sub graphs to a single graph
        dot.subgraph(input_graph)
        dot.subgraph(hidden_graph)
        dot.subgraph(output_graph)
        # Link the nodes based on the connection gene.
        for connection in self.connections:
            dot.edge(connection.source.uid, connection.target.uid,
                     label=str(round(connection.weight, 2)), style='solid')
        return dot
```


```python
gen_simple_network().show()
```




    
![svg](13/output_19_0.svg)
    



Now that we have our network structure in place, we can implement the network activation that takes an input signal as an argument and outputs the result of the network after it has been activated. We can realise a network's activation by computing our neurons' activation values sequentially from the input to the output layer.


```python
class Network(Network):
    def activate(self, inputs: list[float]) -> list[float]:
        # Reset neuron values from previous executions
        for neuron_layer in self.layers.values():
            for neuron in neuron_layer:
                neuron.signal_value = 0

        # Load input features into input neurons
        input_neurons = self.layers.get(0)
        for i in range(len(inputs)):
            input_neurons[i].signal_value = inputs[i]
            input_neurons[i].activate()

        # Traverse the neurons of our network sequentially starting from the hidden layer.
        layer_keys = list(self.layers.keys())
        layer_keys.sort()
        for layer in layer_keys:
            for neuron in self.layers.get(layer):
                if layer > 0:
                    # Calculate weighted sum of incoming connections
                    for connection in neuron.incoming_connections:
                        neuron.signal_value += connection.source.activation_value * connection.weight
                neuron.activate()

        output_neurons = self.layers.get(1)
        return [o.activation_value for o in output_neurons]
```


```python
gen_simple_network().activate([0, 1])
```




    [0.28448328933688793,
     -0.425671744102311,
     -0.2400800088573374,
     0.07554927426248673,
     -0.42763438254010366]



## Inverted Pendulum Problem

We will use our networks to solve the inverted pendulum (= pole-balancing) problem, a well-known benchmark task in the reinforcement learning community. In this problem scenario, a pole is centred on a cart that can be moved to the right and left. Obviously, any movement to the cart also impacts the pole. The task is to move the cart to the left and right such that the pole remains balanced. Whenever the cart position exceeds the boundaries of the track or the pole tips over 12 degrees, the balancing attempt is deemed a failure.

We can simulate the cart and pole system using the following two equations that describe the acceleration of the pole $\ddot{\theta_t}$ and the cart $\ddot{p_t}$ at a given point in time $t$:

$$
\ddot{p}_t = \frac{F_t + m_pl(\dot{\theta^2_t} sin\theta_t - \ddot{\theta_t} \; cos\theta_t)}{m} \; \; \; \; \; \; \; \; (1) \\
\ddot{\theta}_t = \frac{mg \; sin\theta_t - cos\theta_t(F_t + m_pl\dot{\theta}^2_t \; sin\theta_t)}{\frac{4}{3}ml - m_pl \; cos^2(\theta_t)} \; \; \; \; \; \; \; \; (2)
$$


where
- $p$: Position of the cart
- $\dot{p}$: Velocity of the cart
- $\ddot{p}$: Acceleration of the cart
- $\theta$: Angle of the pole
- $\dot{\theta}$: Angular velocity of the pole
- $\ddot{\theta}$: Angular acceleration of the pole
- $l$: Length of the pole
- $m_p$: Mass of the pole
- $m$: Mass of the pole and cart
- $F$: Force applied to the cart
- $g$: Gravity acceleration

In our simulation, we will update both systems using the discrete-time equations

$$
p(t+1) = p(t) + r \dot{p}(t) \; (3) \\
\theta(t+1) = \theta(t) + r \dot{\theta}(t) \;(4) \\
\dot{p}(t+1) = \dot{p}(t) + r \ddot{p}(t) \; (5) \\
\dot{\theta}(t+1) = \dot{\theta}(t) + r \ddot{\theta}(t) \; (6)
$$

with the discrete time step $r$ set to 0.02 seconds.


```python
GRAVITY = 9.8  # m/s^2
MASS_CART = 1.0  # kg
MASS_POLE = 0.1  # kg
TOTAL_MASS = (MASS_CART + MASS_POLE)  # kg
POLE_LENGTH = 0.5  # m
POLEMASS_LENGTH = (MASS_POLE * POLE_LENGTH)
FORCE = 10  # N
STEP_SIZE = 0.02  # sec
FOURTHIRDS = 1.333333


def cart_pole_step(action: int, p: float, p_vel: float, theta: float, theta_vel: float) -> (float, float, float, float):
    force_dir = FORCE if action > 0 else -FORCE

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    temp = (force_dir + POLEMASS_LENGTH * theta_vel * theta_vel * sin_theta) / TOTAL_MASS

    # Equation 2
    theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
                POLE_LENGTH * (FOURTHIRDS - MASS_POLE * cos_theta * cos_theta / TOTAL_MASS))
    # Equation 1
    p_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

    # Compute new states
    p = p + STEP_SIZE * p_vel  # Equation 3
    theta = theta + STEP_SIZE * theta_vel  # Equation 4
    p_vel = p_vel + STEP_SIZE * p_acc  # Equation 5
    theta_vel = theta_vel + STEP_SIZE * theta_acc  # Equation 6

    return p, p_vel, theta, theta_vel
```

Let's test our cart pole system by executing a few steps starting from a clean state where all state variables are set to zero. Given a balanced pole ($\theta = 0$), we expect the pole to always move in the opposite direction of the applied force. Moreover, we expect increasing velocity values as long as we apply a force in the same direction and a decrease in velocity as soon as the force direction changes.


```python
p = 0
p_vel = 0
theta = 0
theta_vel = 0
# Move the cart for 10 steps to the right.
for i in range(10):
    [p, p_vel, theta, theta_vel] = cart_pole_step(1, p, p_vel, theta, theta_vel)
    print(f"Iteration right: {i}\nCart Pos: {p}\nCart Vel: {p_vel}\nPole Angle: {theta}\nPole AVel: {theta_vel}")
    print("--------------------")
          
# Move the cart for 15 steps to the left. 
for i in range(15):
    [p, p_vel, theta, theta_vel] = cart_pole_step(-1, p, p_vel, theta, theta_vel)
    print(f"Iteration left: {i}\nCart Pos: {p}\nCart Vel: {p_vel}\nPole Angle: {theta}\nPole AVel: {theta_vel}")
    print("--------------------")
```

    Iteration right: 0
    Cart Pos: 0.0
    Cart Vel: 0.1951219547888171
    Pole Angle: 0.0
    Pole AVel: -0.29268300535397695
    --------------------
    Iteration right: 1
    Cart Pos: 0.0039024390957763423
    Cart Vel: 0.3902439095776342
    Pole Angle: -0.005853660107079539
    Pole AVel: -0.5853660107079539
    --------------------
    Iteration right: 2
    Cart Pos: 0.011707317287329027
    Cart Vel: 0.5854473662672043
    Pole Angle: -0.017560980321238616
    Pole AVel: -0.8798872190960108
    --------------------
    Iteration right: 3
    Cart Pos: 0.023416264612673113
    Cart Vel: 0.7808034482987078
    Pole Angle: -0.03515872470315883
    Pole AVel: -1.178038896588744
    --------------------
    Iteration right: 4
    Cart Pos: 0.03903233357864727
    Cart Vel: 0.9763639418404654
    Pole Angle: -0.05871950263493371
    Pole AVel: -1.4815329625443723
    --------------------
    Iteration right: 5
    Cart Pos: 0.05855961241545658
    Cart Vel: 1.172151077414685
    Pole Angle: -0.08835016188582115
    Pole AVel: -1.7919612011435535
    --------------------
    Iteration right: 6
    Cart Pos: 0.08200263396375028
    Cart Vel: 1.3681454441981398
    Pole Angle: -0.12418938590869222
    Pole AVel: -2.1107473332066196
    --------------------
    Iteration right: 7
    Cart Pos: 0.10936554284771308
    Cart Vel: 1.5642715914153589
    Pole Angle: -0.1664043325728246
    Pole AVel: -2.4390888082002777
    --------------------
    Iteration right: 8
    Cart Pos: 0.14065097467602025
    Cart Vel: 1.76038115934149
    Pole Angle: -0.21518610873683017
    Pole AVel: -2.7778872734265025
    --------------------
    Iteration right: 9
    Cart Pos: 0.17585859786285005
    Cart Vel: 1.956233861883977
    Pole Angle: -0.27074385420536023
    Pole AVel: -3.127668490894963
    --------------------
    Iteration left: 0
    Cart Pos: 0.21498327510052959
    Cart Vel: 1.7632652933902342
    Pole Angle: -0.3332972240232595
    Pole AVel: -2.9273895035470296
    --------------------
    Iteration left: 1
    Cart Pos: 0.25024858096833424
    Cart Vel: 1.571344914406432
    Pole Angle: -0.3918450140942001
    Pole AVel: -2.7515365246297137
    --------------------
    Iteration left: 2
    Cart Pos: 0.28167547925646286
    Cart Vel: 1.3805005689274916
    Pole Angle: -0.44687574458679435
    Pole AVel: -2.5992441491311347
    --------------------
    Iteration left: 3
    Cart Pos: 0.3092854906350127
    Cart Vel: 1.1907126899876923
    Pole Angle: -0.49886062756941707
    Pole AVel: -2.469569739043521
    --------------------
    Iteration left: 4
    Cart Pos: 0.33309974443476653
    Cart Vel: 1.0019309505127758
    Pole Angle: -0.5482520223502875
    Pole AVel: -2.3615648963925833
    --------------------
    Iteration left: 5
    Cart Pos: 0.35313836344502203
    Cart Vel: 0.8140860856326241
    Pole Angle: -0.5954833202781391
    Pole AVel: -2.274325969546335
    --------------------
    Iteration left: 6
    Cart Pos: 0.3694200851576745
    Cart Vel: 0.6270978571284449
    Pole Angle: -0.6409698396690658
    Pole AVel: -2.2070281641767515
    --------------------
    Iteration left: 7
    Cart Pos: 0.3819620423002434
    Cart Vel: 0.44088004947248216
    Pole Angle: -0.6851104029526008
    Pole AVel: -2.1589473496805938
    --------------------
    Iteration left: 8
    Cart Pos: 0.39077964328969306
    Cart Vel: 0.2553432345191425
    Pole Angle: -0.7282893499462127
    Pole AVel: -2.1294729185112797
    --------------------
    Iteration left: 9
    Cart Pos: 0.3958865079800759
    Cart Vel: 0.07039587671876138
    Pole Angle: -0.7708788083164384
    Pole AVel: -2.1181143011627457
    --------------------
    Iteration left: 10
    Cart Pos: 0.39729442551445115
    Cart Vel: -0.11405579905980248
    Pole Angle: -0.8132410943396933
    Pole AVel: -2.1245030706087547
    --------------------
    Iteration left: 11
    Cart Pos: 0.3950133095332551
    Cart Vel: -0.29810887869795544
    Pole Angle: -0.8557311557518683
    Pole AVel: -2.1483920226503734
    --------------------
    Iteration left: 12
    Cart Pos: 0.38905113195929597
    Cart Vel: -0.4818655664058769
    Pole Angle: -0.8986989962048758
    Pole AVel: -2.189652192573561
    --------------------
    Iteration left: 13
    Cart Pos: 0.37941382063117846
    Cart Vel: -0.6654355960582027
    Pole Angle: -0.942492040056347
    Pole AVel: -2.2482684480876864
    --------------------
    Iteration left: 14
    Cart Pos: 0.3661051087100144
    Cart Vel: -0.8489391605252926
    Pole Angle: -0.9874574090181008
    Pole AVel: -2.324334063461633
    --------------------


We can formulate our pole-balancing task as a reinforcement learning problem by implementing a simulation in which an agent's goal is to balance the pole for as long as possible. In our scenario, our agents are neural networks that decide in which direction the cart should be moved at a given state $[p, \theta, \dot{p}, \dot{\theta}]$. Since we have four input features, the networks will have four input neurons. Furthermore, these input features will be normalised over the following ranges:

$$
p: [-2.4, 2.4] \\
\dot{p}: [-1.5, 1.5] \\
\theta: [-12, 12] \\
\dot{\theta}: [-60, 60] \\
$$

We model the pole-balancing problem as a binary classification task using two output neurons, each representing one action. An action is chosen by selecting the action that belongs to the neuron with the highest activation value. Once the pole tips over 12 degrees or the 4.8 meter track is left by the cart, the trial ends. Finally, an agent's performance is measured by the number of steps it managed to survive. A balancing attempt is deemed successful whenever an agent balances the pole for 120,000 time steps, which is equivalent to 40 minutes in real time.

For a more challenging second scenario, we add an option that randomises the starting positions of the four input states.


```python
MAX_STEPS = 120000
TWELVE_DEGREES = 0.2094395  #conversion to rad (12*pi)/180

def evaluate_agent(network: Network, random_start=False) -> int:
    # Define starting state
    if random_start:
        p = random.uniform(-2.4, 2.4)  # -2.4 < p < 2.4
        p_vel = random.uniform(-1.5, 1.5)  # -1.5 < p_acc < 1.5
        theta = random.uniform(-TWELVE_DEGREES, TWELVE_DEGREES)  # -12 < theta < 12
        theta_vel = random.uniform(-1, 1)  # -60 < theta_acc < 60 (in rad)
    else:
        p, p_vel, theta, theta_vel = 0.0, 0.0, 0.0, 0.0

    # Simulation loop
    steps = 0
    while steps < MAX_STEPS:
        # Normalise inputs
        inputs = [None] * 5
        inputs[0] = (p + 2.4) / 4.8
        inputs[1] = (p_vel + 1.5) / 3
        inputs[2] = (theta + TWELVE_DEGREES) / (2 * TWELVE_DEGREES)
        inputs[3] = (theta_vel + 1.0) / 2.0
        inputs[4] = 0.5

        # Activate the network and interpret the output, and advance the state.
        net_output = network.activate(inputs)
        action = -1 if net_output[0] > net_output[1] else 1
        p, p_vel, theta, theta_vel = cart_pole_step(action, p, p_vel, theta, theta_vel)

        # Check if the attempt is still valid
        if p < -2.4 or p > 2.4 or theta < -TWELVE_DEGREES or theta > TWELVE_DEGREES:
            return steps

        steps += 1

    # At this point the agent survived for the maximum number of steps
    return steps
```

Let's create a sample network and test it in the pole-balancing environment.


```python
net = gen_simple_network()
evaluate_agent(net, random_start=False)
```




    8



As expected, our randomly generated networks could perform better. Therefore, we will now implement a Neuroevolution algorithm to optimise our networks and solve the pole-balancing task.

## SANE (Symbiotic Adaptive Neuro-Evolution)

[SANE](https://link.springer.com/article/10.1023/A:1018004120707) implements a symbiotic evolution approach, which means that it does not evolve complete solutions but parts thereof. In the case of SANE, we maintain a population of hidden neurons, which are then combined to form a neural network based on a pre-defined input and output layer. Symbiotic approaches aim to avoid premature convergence by enforcing diverse populations and specialisation of individuals in specific subtasks within the problem environment. Each individual receives a fitness score based on the average fitness of the networks in which they participated.

### Genetic Representation

Each neuron is defined by a label and weight list. The former defines the neuron from/to which the connection is coming/going, and the latter the corresponding link weight. We follow the original SANE implementation of Moriarty and Mikkulainen and define an upper bound of 254 for our label values. We connect from the input layer to the hidden neuron if the label value is smaller than 127, and otherwise, from the hidden neuron to the output layer. Moreover, we apply modulo calculus over the total number of input/output neurons to decide to which specific input/output neuron a connection is created.


```python
LABEL_BOUND = 254


class HiddenNeuronGene:
    hidden_neuron_counter = 0

    def __init__(self, labels: list[int], weights: list[float]):
        self.labels = [l % LABEL_BOUND for l in labels]  # Enforce upper label boundary.
        self.weights = weights
        self.fitness = 0
        self.evaluations = 0
        self.hidden_neuron = HiddenNeuron("H" + str(HiddenNeuronGene.hidden_neuron_counter))
        HiddenNeuronGene.hidden_neuron_counter += 1

    def generate_connections(self, input_neurons: list[Neuron], output_neurons: list[Neuron]) -> list[Connection]:
        connections = []
        for label, weight in zip(self.labels, self.weights):

            # Generate connection to input layer
            if label < 127:
                neuron_id = "I" + str(label % len(input_neurons))
                in_neuron = next((i for i in input_neurons if i.uid == neuron_id), None)
                connections.append(Connection(in_neuron, self.hidden_neuron, weight))

            # Generate connection to output layer
            else:
                neuron_id = "O" + str(label % len(output_neurons))
                out_neuron = next((o for o in output_neurons if o.uid == neuron_id), None)
                connections.append(Connection(self.hidden_neuron, out_neuron, weight))

        return connections
```

Let's test our gene implementation by generating a neural network using three randomly generated HiddenNeuronGenes, each realising five connections. Moreover, we implement a helper function to create chromosomes with random gene values.


```python
def generate_random_gene_sane(num_connections: int) -> HiddenNeuronGene:
    labels = []
    weights = []
    for _ in range(num_connections):
        labels.append(math.floor(random.uniform(0, LABEL_BOUND)))
        weights.append(random.uniform(-1, 1))
    return HiddenNeuronGene(labels, weights)


def generate_network_SANE(chromosomes: list[HiddenNeuronGene]) -> Network:
    input_neurons = [InputNeuron(f"I{x}") for x in range(5)]
    output_neurons = [OutputNeuron(f"O{x}") for x in range(2)]

    # Extract the hidden neurons and their connections from the provided chromosomes.
    hidden_neurons = []
    connections = []
    for chromosome in chromosomes:
        hidden_neurons.append(chromosome.hidden_neuron)
        connections.extend(chromosome.generate_connections(input_neurons, output_neurons))

    # Assemble in layers.
    layers = {0: input_neurons, 0.5: hidden_neurons, 1: output_neurons}

    return Network(layers, connections)
```


```python
genes = [generate_random_gene_sane(5) for x in range(3)]
net = generate_network_SANE(genes)
net.show()
```




    
![svg](13/output_41_0.svg)
    



### Crossover

SANE primarily breeds offspring using a single-point crossover operator.


```python
def crossover_sane(parent_1: HiddenNeuronGene, parent_2: HiddenNeuronGene) -> [HiddenNeuronGene, HiddenNeuronGene]:
    # Extract the genes from both parents
    label_gene_1, weight_gene_1 = parent_1.labels, parent_1.weights
    label_gene_2, weight_gene_2 = parent_2.labels, parent_2.weights

    # Determine the crossover point. We assume both parent genes have the same gene lengths.
    crossover_point = math.floor(random.uniform(1, len(label_gene_1)))

    # Apply the crossover operation
    child_1_labels = label_gene_1[:crossover_point] + label_gene_2[crossover_point:]
    child_1_weights = weight_gene_1[:crossover_point] + weight_gene_2[crossover_point:]
    child_2_labels = label_gene_2[:crossover_point] + label_gene_1[crossover_point:]
    child_2_weights = weight_gene_2[:crossover_point] + weight_gene_1[crossover_point:]

    # Breed and return the two children.
    return [HiddenNeuronGene(child_1_labels, child_1_weights), HiddenNeuronGene(child_2_labels, child_2_weights)]
```


```python
parent_1 = generate_random_gene_sane(5)
parent_2 = generate_random_gene_sane(5)
parent_1.weights = [round(w * 100) / 100 for w in parent_1.weights]  # Round weights for better visualisation
parent_2.weights = [round(w * 100) / 100 for w in parent_2.weights]
[child_1, child_2] = crossover_sane(parent_1, parent_2)

print("Neuron \t\t| Labels \t\t\t| Weights")
print("--------------------------------------------------------------------------------")
print(f"Parent 1 \t| {parent_1.labels} \t| {parent_1.weights}")
print(f"Parent 2 \t| {parent_2.labels} \t| {parent_2.weights}")
print(f"Child 1 \t| {child_1.labels} \t| {child_1.weights}")
print(f"Child 2 \t| {child_2.labels} \t| {child_2.weights}")
```

    Neuron 		| Labels 			| Weights
    --------------------------------------------------------------------------------
    Parent 1 	| [240, 211, 5, 128, 37] 	| [-0.57, -0.82, -0.45, 0.93, 0.37]
    Parent 2 	| [225, 142, 229, 32, 170] 	| [0.12, 0.65, 0.87, -0.07, -0.68]
    Child 1 	| [240, 211, 229, 32, 170] 	| [-0.57, -0.82, 0.87, -0.07, -0.68]
    Child 2 	| [225, 142, 5, 128, 37] 	| [0.12, 0.65, -0.45, 0.93, 0.37]


### Mutation

The mutation operator adds gaussian noise to the values of the two gene lists.
 
In contrast, to other evolutionary algorithms, SANE uses mutation only to add genetic material that may be missing from the initial population. Thus, mutation is not responsible for creating diversity within the population as diversity is already present due to the symbiotic evolution approach.


```python
MUTATION_POWER = 2


def mutate_sane(parent: HiddenNeuronGene) -> HiddenNeuronGene:
    mutated_labels = []
    mutated_weights = []
    # Add gaussian noise to the values.
    for label, weight in zip(parent.labels, parent.weights):
        mutated_labels.append(math.floor(np.random.normal(label, MUTATION_POWER)))
        mutated_weights.append(np.random.normal(weight, MUTATION_POWER / 10))

    return HiddenNeuronGene(mutated_labels, mutated_weights)
```


```python
parent = generate_random_gene_sane(5)
mutant = mutate_sane(parent)
parent.weights = [round(w * 100) / 100 for w in parent.weights]  # Round weights for better visualisation
mutant.weights = [round(w * 100) / 100 for w in mutant.weights]
print("Neuron \t|\t\t Labels \t\t|\t\t Weights")
print("-------------------------------------------------------------------------------------")
print(f"Parent \t|\t {parent.labels} \t|\t {parent.weights}")
print(f"Mutant \t|\t {mutant.labels} \t|\t {mutant.weights}")
```

    Neuron 	|		 Labels 		|		 Weights
    -------------------------------------------------------------------------------------
    Parent 	|	 [2, 75, 182, 17, 154] 	|	 [-0.77, -0.12, 0.8, -0.77, -0.48]
    Mutant 	|	 [3, 74, 178, 17, 150] 	|	 [-0.48, -0.28, 0.79, -1.03, -0.84]


### Evolution

A new generation is formed by applying crossover to the best-performing members of the current population. Parents are selected sequentially, and mating partners must have a fitness value at least as high as the selected parent. Finally, we apply mutation to fill the remaining slots in our population.


```python
NUM_PARENTS = 50
POPULATION_SIZE = 200


def evolve_sane(population: list[HiddenNeuronGene]) -> HiddenNeuronGene:
    # Sort the population such that the best-performing chromosomes are at the beginning.
    population.sort(key=lambda x: x.fitness, reverse=True)
    offspring = []

    # Crossover
    for i in range(NUM_PARENTS):
        first_parent = population[i]
        second_parent = population[math.floor(random.uniform(0, i))]
        offspring.extend(crossover_sane(first_parent, second_parent))

    # Mutation
    while len(offspring) < POPULATION_SIZE:
        offspring.append(mutate_sane(random.choice(population)))

    return offspring
```


```python
# Quick sanity check
population = [generate_random_gene_sane(5) for i in range(POPULATION_SIZE)]
offspring = evolve_sane(population)
print(f"Parent Population Size: {len(population)}")
print(f"Offspring Population Size: {len(offspring)}")
```

    Parent Population Size: 200
    Offspring Population Size: 200


### Fitness evaluation

Since our population consists of partial solutions and not full-fledged neural networks, we have to combine several chromosomes into a neural network for each evaluation episode. The fitness of a single chromosome is set to the average fitness of all networks in which it participated. For our pole-balancing task, we combine eight chromosomes, each realising five connections. Hence, we obtain networks that have a total of 40 connections. In each iteration, we generate 200 networks while maintaining a neuron population size of 200, which means that, on average, each neuron is allowed to participate in eight networks per generation.


```python
NUM_NETWORKS, POPULATION_SIZE, NUM_HIDDEN_NEURONS, NUM_CONNECTIONS = 200, 200, 8, 5

def evaluate_population_sane(population: list[HiddenNeuronGene], evaluations: int, randomise=False) -> (int, int):
    for c in population:     # Reset fitness and evaluation counter for each neuron
        c.fitness = 0
        c.evaluations = 0

    best = 0
    for i in range(NUM_NETWORKS):      # Perform NUM_NETWORKS evaluation rounds.
        hidden_neurons = []
        random.shuffle(population)
        for i in range(NUM_HIDDEN_NEURONS):  # Combine NUM_HIDDEN_NEURONS in a network.
            candidate = population[i]
            candidate.evaluations += 1
            hidden_neurons.append(candidate)
            
        network = generate_network_SANE(hidden_neurons) # Generate and evaluate networks.
        fitness = evaluate_agent(network, randomise)
        evaluations += 1
        
        # Assign the obtained fitness values to the chromosomes.
        for neuron in hidden_neurons:
            neuron.fitness += fitness

        if fitness > best:
            best = fitness

        if fitness == MAX_STEPS:
            return best, evaluations

    # Compute the final fitness values by averaging over the number of evaluations
    for individual in population:
        if individual.evaluations > 0:
            individual.fitness = individual.fitness / individual.evaluations
        else:
            individual.fitness = 0

    return best, evaluations
```


```python
population = [generate_random_gene_sane(NUM_CONNECTIONS) for i in range(POPULATION_SIZE)]
evaluate_population_sane(population, 0)
```




    (171, 200)



### Solving the Pole-Balancing Task - SANE

With everything in place, we can finally implement our neuroevolution algorithm to solve the pole-balancing problem.


```python
NUM_NETWORKS, POPULATION_SIZE, NUM_HIDDEN_NEURONS, NUM_CONNECTIONS = 200, 200, 8, 5
MUTATION_RATE, MUTATION_POWER = 0.1, 2
NUM_PARENTS, MAX_GENERATIONS = 50, 1000

def solve_pole_sane(random_starts: bool, print_log: bool) -> (list[int], int):
    # Initialise population
    population = [generate_random_gene_sane(NUM_CONNECTIONS) for _ in range(POPULATION_SIZE)]

    num_generation = 0
    evaluations = 0
    fitness_values = []
    while num_generation < MAX_GENERATIONS:
        best_fitness, evaluations = evaluate_population_sane(population, evaluations, random_starts)
        fitness_values.append(best_fitness)

        if best_fitness == MAX_STEPS:
            print(f"SANE found solution in generation {num_generation} after {evaluations} evaluations")
            break

        if print_log:
            print(f"Generation {num_generation}")
            print(f"Best fitness {best_fitness}")

        population = evolve_sane(population)
        num_generation += 1
    return fitness_values, evaluations
```


```python
fitness_values, _ = solve_pole_sane(False, False)
x_range = np.arange(0, len(fitness_values), 1)

plt.plot(x_range, fitness_values)
plt.yscale("log")
```

    SANE found solution in generation 7 after 1418 evaluations



    
![png](13/output_56_1.png)
    


### Evaluation of SANE

For both scenarios (static and random start), we repeat the experiment a couple of times and measure the average fitness of the best-performing network over time.


```python
def evaluate_sane(random_start: bool, repetitions: int) -> (list[int], list[int]):
    fitness_vals = []
    evaluations = []
    for _ in range(repetitions):
        fitness, evals = solve_pole_sane(random_start, False)
        fitness_vals.append(fitness)
        evaluations.append(evals)

    return fitness_vals, evaluations


def plot(fitness_values: list[list[int]]):
    # Padding
    max_generation = max([len(f) for f in fitness_values])
    for f in fitness_values:
        while len(f) < max_generation:
            f.append(MAX_STEPS)

    mean_fitness_time = np.array(fitness_values).mean(axis=0)
    x_range = np.arange(0, max_generation, 1)
    plt.plot(x_range, mean_fitness_time)
    plt.yscale("log") 
```


```python
repetitions = 5
random_start = True
fitness, evaluation = evaluate_sane(random_start, repetitions)
plot(fitness)
print(f"Average number of evaluations SANE: {mean(evaluation)}\n")
```

    SANE found solution in generation 43 after 8613 evaluations
    SANE found solution in generation 4 after 835 evaluations
    SANE found solution in generation 3 after 613 evaluations
    SANE found solution in generation 39 after 7821 evaluations
    SANE found solution in generation 64 after 12978 evaluations
    Average number of evaluations SANE: 6172
    



    
![png](13/output_59_1.png)
    


## CoSyNE (Cooperative Synapse Neuroevolution)

[CoSyNE](https://nn.cs.utexas.edu/downloads/papers/gomez.jmlr08.pdf) evolves connection weights by encoding all connection weights $m$ of a fixed topology network into subgenes that are distributed over $m$ species. Since each species maintains $n$ genes, we can model our population by a $n$ x $m$ matrix. A functional network can be generated by combining a single gene of each species within a single row of the matrix. 

The most crucial feature of CoSyNE is the cooperative evolution approach realised by permuting the rows of a single subpopulation, leading to new combinations of the subgenes and new networks.

### Genetic Representation

We assign each subgene of CoSyNE a weight and fitness value. The former defines the weight for a corresponding connection and the latter the achieved fitness of the network in which the gene participated. Finally, each gene has a permutation attribute that defines the probability of being permutated in the cooperative evolution step.


```python
class WeightGene:

    def __init__(self, weight, fitness=0):
        self.weight = weight
        self.fitness = fitness
        self.permutation_prob = 0
```

### Generating the Initial Population

We can create a new population by generating $n$ x $m$ weight genes having a random weight value. 


```python
WEIGHT_MAGNITUDE = 1


def init_population_COSYNE(num_weights: int, subpop_size: int) -> list[list[WeightGene]]:
    population = []
    for i in range(subpop_size): # Number of weights per species == Number of Networks in Population
        single_network = []
        for j in range(num_weights): # Number of Species == Number of weights per Network
            single_network.append(WeightGene(random.uniform(-WEIGHT_MAGNITUDE, WEIGHT_MAGNITUDE)))
        population.append(single_network)
    return population
```

To visualise our population, we implement a helper function that takes a population as input and outputs the respective weights.


```python
def show_population_COSYNE(population: list[list[WeightGene]]) -> list[list[float]]:
    weights = []
    for network in population:
        network_weights = []
        for weight_gene in network:
            network_weights.append(weight_gene.weight)
        weights.append(network_weights)
    return weights


pop = init_population_COSYNE(2, 4)
show_population_COSYNE(pop)
```




    [[0.8698148532730281, 0.9054056122911787],
     [-0.05055385104728205, -0.9191048319160044],
     [0.27442788921291617, 0.791290916783737],
     [-0.4853344289304362, -0.2759418290443414]]



### Weight Genes --> Networks

Since we have to transform our weights into networks for the evaluation and evolution phase, we implement a function that converts a list of weight genes to the corresponding network. Moreover, because CoSyNE evolves fixed topologies, we have to define the network topology by specifying the number of input, hidden and output neurons.


```python
class Network(Network):
    def __init__(self, layers: dict[float, list[Neuron]], connections: list[Connection]):
        self.layers = layers
        self.connections = connections
        self.generate()
        self.fitness = 0
```


```python
def weights_to_network(weight_genes: list[WeightGene], num_in: int, num_hidden: int, num_out: int) -> Network:
    input_neurons = [InputNeuron(f"I{x}") for x in range(num_in)]
    hidden_neurons = [HiddenNeuron(f"H{x}") for x in range(num_hidden)]
    output_neurons = [OutputNeuron(f"O{x}") for x in range(num_out)]

    # Check number of connections
    required_connections = num_in * num_hidden + num_hidden * num_out
    assert len(weight_genes) == required_connections

    # Create connections
    weight_index = 0
    connections = []
    for in_neuron in input_neurons:
        for hidden_neuron in hidden_neurons:
            connections.append(Connection(in_neuron, hidden_neuron, weight_genes[weight_index].weight))
            weight_index += 1
    for hidden_neuron in hidden_neurons:
        for out_neuron in output_neurons:
            connections.append(Connection(hidden_neuron, out_neuron, weight_genes[weight_index].weight))
            weight_index += 1

    layers = {0: input_neurons, 0.5: hidden_neurons, 1: output_neurons}
    network = Network(layers, connections)
    network.fitness = weight_genes[0].fitness  # Fitness of all genes is the same
    return network
```


```python
weights = init_population_COSYNE(12, 1)
network = weights_to_network(weights[0], 5, 2, 1)
network.show()
```




    
![svg](13/output_70_0.svg)
    



Based on the previous function, we can now transform a whole CoSyNE population into the respective networks by applying the **weights_to_network** function to each row of our population matrix.


```python
def weights_to_networks_pop(weights: list[list[WeightGene]], num_in: int, num_hidden: int,
                            num_out: int) -> list[Network]:
    return [weights_to_network(w, num_in, num_hidden, num_out) for w in weights]


weights = init_population_COSYNE(3, 10)
networks = weights_to_networks_pop(weights, 2, 1, 1)

print(f"Networks vs. Subpopulations: {len(networks)} ==> {len(weights)}")
print(f"Connections vs. Num Weights: {len(networks[0].connections)} ==> {len(weights[0])}")
weight_values = [w.weight for w in weights[0]]
connection_values = [c.weight for c in networks[0].connections]
print("\nConnection Weights vs. Weight Values:")
print(f"{weight_values}\n{connection_values}")
```

    Networks vs. Subpopulations: 10 ==> 10
    Connections vs. Num Weights: 3 ==> 3
    
    Connection Weights vs. Weight Values:
    [0.2606544444789465, 0.0884618915846398, -0.40453239554009635]
    [0.2606544444789465, 0.0884618915846398, -0.40453239554009635]


### Networks ---> Weight Genes

In order to facilitate the mapping between network and subgene fitness values, we extend the attributes of our networks with a fitness value.

We implement the complement to the former two conversion functions such that we can extract weight genes, including their fitness values, from the respective networks.


```python
def network_to_weights(network: Network) -> list[WeightGene]:
    weight_genes = []
    for connection in network.connections:
        weight_genes.append(WeightGene(connection.weight, network.fitness))
    return weight_genes


weights = init_population_COSYNE(3, 1)
network = weights_to_network(weights[0], 2, 1, 1)
net_weights = network_to_weights(network)

print(show_population_COSYNE(weights))
print(show_population_COSYNE([net_weights]))
```

    [[-0.3627429722916766, 0.0263674055649068, -0.896150950593871]]
    [[-0.3627429722916766, 0.0263674055649068, -0.896150950593871]]



```python
def networks_to_weights_pop(networks: list[Network]) -> list[list[WeightGene]]:
    weight_population = []
    for network in networks:
        network_weights = []
        weight_population.append(network_to_weights(network))
    return weight_population


weights = init_population_COSYNE(3, 1)
networks = weights_to_networks_pop(weights, 2, 1, 1)
network_weights = networks_to_weights_pop(networks)

print(f"Subpopulation size before/after: {len(weights)} ==> {len(network_weights)}")
print(f"Weight size before/after: {len(weights[0])} ==> {len(network_weights[0])}")
weight_values_before = [w.weight for w in weights[0]]
weight_values_after = [w.weight for w in network_weights[0]]
print(f"Weight values before/after: \n{weight_values_before}\n{weight_values_after}")
```

    Subpopulation size before/after: 1 ==> 1
    Weight size before/after: 3 ==> 3
    Weight values before/after: 
    [-0.6406688571723775, -0.2139793436652857, -0.35954405878728424]
    [-0.6406688571723775, -0.2139793436652857, -0.35954405878728424]


### Crossover

CoSyNE applies multi-point crossover, where each weight gene and its respective weight value is inherited randomly by one of the two parents. We use a mask consisting of zeroes and ones and multiply the weights of the first parent elementwise with the mask. Then we add the weights of the second parent after multiplying it with the inverted mask. Finally, the second child is produced by switching the parents' positions.


```python
def crossover_cosyne(parent_1: Network, parent_2: Network) -> (Network, Network):
    weights_p1 = np.array([w.weight for w in network_to_weights(parent_1)])
    weights_p2 = np.array([w.weight for w in network_to_weights(parent_2)])

    # Crossover mask decides which elements will be switched
    crossover_mask = np.array([1 if random.uniform(0, 1) < 0.5 else 0 for _ in range(len(weights_p1))])

    # Form children via element-wise multiplication of crossover_mask and elementwise complement addition with mating parent.
    weights_c1 = weights_p1 * crossover_mask + weights_p2 * (1 - crossover_mask)
    weights_c2 = weights_p2 * crossover_mask + weights_p1 * (1 - crossover_mask)

    weight_genes_c1 = [WeightGene(w) for w in weights_c1]
    weight_genes_c2 = [WeightGene(w) for w in weights_c2]

    num_in = len(parent_1.layers.get(0))
    num_hidden = len(parent_1.layers.get(0.5))
    num_out = len(parent_1.layers.get(1))

    c1 = weights_to_network(weight_genes_c1, num_in, num_hidden, num_out)
    c2 = weights_to_network(weight_genes_c2, num_in, num_hidden, num_out)
    return c1, c2
```


```python
num_in = 2
num_hidden = 1
num_out = 1
num_weights = num_in * num_hidden + num_hidden * num_out

weights = init_population_COSYNE(num_weights, 2)
network_1 = weights_to_network(weights[0], num_in, num_hidden, num_out)
network_2 = weights_to_network(weights[1], num_in, num_hidden, num_out)
child_1, child_2 = crossover_cosyne(network_1, network_2)

print(f"Parent 1: {[w.weight for w in weights[0]]}")
print(f"Parent 2: {[w.weight for w in weights[1]]}")
print(f"Child 1: {[w.weight for w in network_to_weights(child_1)]}")
print(f"Child 2: {[w.weight for w in network_to_weights(child_2)]}")
```

    Parent 1: [0.6781938392241649, 0.38838115026585984, 0.8742702392985826]
    Parent 2: [-0.34854345384889274, 0.5133450470470633, -0.8726119532756935]
    Child 1: [np.float64(0.6781938392241649), np.float64(0.5133450470470633), np.float64(-0.8726119532756935)]
    Child 2: [np.float64(-0.34854345384889274), np.float64(0.38838115026585984), np.float64(0.8742702392985826)]


### Mutation

Mutation is applied on the network level and adds noise to the connection weights based on a predefined mutation probability.


```python
COSYNE_MUTATION_RATE = 0.3
COSYNE_MUTATION_POWER = 1


def mutate_cosyne(networks: list[Network]):
    for network in networks:
        for connection in network.connections:
            if random.uniform(0, 1) < COSYNE_MUTATION_RATE:
                connection.weight = np.random.normal(connection.weight, COSYNE_MUTATION_POWER)


num_in = 2
num_hidden = 1
num_out = 1
num_weights = num_in * num_hidden + num_hidden * num_out

weights = init_population_COSYNE(num_weights, 1)
network = weights_to_network(weights[0], num_in, num_hidden, num_out)
print(f"Parent: {show_population_COSYNE(weights)}")

mutate_cosyne([network])
mutant_weights = network_to_weights(network)
print(f"Mutant: {[w.weight for w in mutant_weights]}")
```

    Parent: [[-0.22415080420875833, 0.22123857905611466, 0.14871365189126684]]
    Mutant: [-0.22415080420875833, 0.12834598544240042, 0.14871365189126684]


### Evolution

A new generation is formed by inheriting the first half of the best-performing networks. The second half is then generated by mating the upper quarter of the best-performing networks. Finally, we mutate all networks in the resulting offspring population.


```python
SUBPOP_SIZE = 100
COSYNE_PARENTS = math.floor(0.25 * SUBPOP_SIZE)
INPUT_SIZE = 5
HIDDEN_SIZE = 3
OUTPUT_SIZE = 2
NUM_COSYNE_CONNECTIONS = INPUT_SIZE * HIDDEN_SIZE + OUTPUT_SIZE * HIDDEN_SIZE


def evolve_cosyne(networks: list[Network]) -> list[Network]:
    # Sort networks by fitness
    networks.sort(key=lambda x: x.fitness, reverse=True)

    # First half based on best-performing networks
    half_length = int(0.5 * len(networks))
    offspring = networks[:half_length]

    # Second half based on children of best-performing networks
    mating_children = []
    for i in range(COSYNE_PARENTS):
        parent_1 = offspring[i]
        parent_2 = random.choice(offspring[:COSYNE_PARENTS])
        mating_children.extend(crossover_cosyne(parent_1, parent_2))

    # Add mating children to population
    offspring.extend(mating_children)

    # Mutate
    mutate_cosyne(networks)

    return networks


weights = init_population_COSYNE(NUM_COSYNE_CONNECTIONS, 100)
networks = weights_to_networks_pop(weights, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

print(f"Old Generation Size: {len(networks)}")
offspring = evolve_cosyne(networks)
print(f"New Generation Size: {len(offspring)}")
```

    Old Generation Size: 100
    New Generation Size: 100


### Coevolution - Permuting The Subgenes

The coevolution principle is realised by permuting the row positions of our subgenes within their respective species. For this purpose, we implement two helper functions for extracting and replacing a single column from our matrix population.


```python
def get_column(index: int, matrix:list[list]):
    return [row[index] for row in matrix]


def replace_column(index: int, matrix:list[list], column:list):
    for i, row in enumerate(matrix):
        row[index] = column[i]


test_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(f"Before modification {get_column(1, test_matrix)}")
replace_column(1, test_matrix, [-1, 0, 1])
print(f"After modification {get_column(1, test_matrix)}")
```

    Before modification [2, 5, 8]
    After modification [-1, 0, 1]


The permutation algorithm starts by computing the permutation probability of each weight gene based on the equation

$$prob(x_{ij}) = 1 - \sqrt[n]{\frac{f(x_{ij})-f_i^{min}}{f_i^{max} - f_i^{min}}}$$

, where $f(x_{ij})$ corresponds to the fitness of individual $x_{ij}$ and $f_i^{min}$/$f_i^{max}$ to the least and most fit individuals of subpopulation $i$. Therefore, the probability of changing a network at position $j$ is inversely proportional to its fitness, such that good-performing networks are more likely to be preserved, and poor networks are recombined to search for better solutions.

Governed by the computed probability, we shift each gene to the nearest neighbour for which randomness determined that permutation should be applied.


```python
def permute(weight_population: list[list[WeightGene]]) -> list[list[WeightGene]]:
    
    # Sort matrix list by fitness of respective networks
    networks = weights_to_networks_pop(weight_population, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    networks.sort(key=lambda x: x.fitness, reverse=True)
    weight_population = networks_to_weights_pop(networks)
    
    for i in range(len(weight_population[0])):   # Iterate over weights in subpopulation
        sub_pop = get_column(i, weight_population)

        # Calculate probability of permutation for each WeightGene
        min_fitness = min([w.fitness for w in sub_pop])
        max_fitness = max([w.fitness for w in sub_pop])
        for gene in sub_pop:
            if max_fitness == min_fitness:
                gene.permutation_prob = 1
            else:
                temp = (gene.fitness - min_fitness) / (max_fitness - min_fitness)
                gene.permutation_prob = 1 - math.pow(temp, 1 / len(sub_pop))

        # Permute based on determined probabilities
        curr = None
        first_index = None
        for j, gene in enumerate(sub_pop):
            if random.uniform(0, 1) < gene.permutation_prob:
                if curr is None:
                    first_index = j
                    curr = gene
                else:
                    tmp = sub_pop[j]
                    sub_pop[j] = curr
                    curr = tmp
                    
        sub_pop[first_index] = curr
        replace_column(i, weight_population, sub_pop)

    return weight_population
```


```python
num_weights = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE
weights = init_population_COSYNE(num_weights, 5)
networks = weights_to_networks_pop(weights, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
weights = networks_to_weights_pop(networks)


subpop_weights = [w.weight for w in get_column(0, weights)]
print(f"Before: {subpop_weights}")
weights_after = permute(weights)
subpop_weights_after = [w.weight for w in get_column(0, weights_after)]
print(f"After: {subpop_weights_after}")
```

    Before: [0.22209375547612265, 0.6272852508399194, 0.1957671918876207, 0.7631603010738093, 0.7404774066895221]
    After: [0.7404774066895221, 0.22209375547612265, 0.6272852508399194, 0.1957671918876207, 0.7631603010738093]


### Population Evaluation

Once again, we will evaluate our alogrithm on the pole-balancing problem.


```python
def evaluate_population_cosyne(networks: list[Network], evaluations: int, random_start=False) -> (int, int):
    best = 0
    for network in networks:
        evaluations += 1
        fitness = evaluate_agent(network, random_start)
        network.fitness = fitness

        if fitness == MAX_STEPS:
            return fitness, evaluations

        if fitness > best:
            best = fitness

    return best, evaluations


weights = init_population_COSYNE(NUM_COSYNE_CONNECTIONS, 100)
networks = weights_to_networks_pop(weights, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
best_fitness, _ = evaluate_population_cosyne(networks, 0)
best_fitness
```




    17



### Solving the Pole-Balancing Task - CoSyNE

Finally, we assemble our evaluation, evolution and permutation functions of CoSyNE to solve the pole balancing task.


```python
INPUT_SIZE = 5
HIDDEN_SIZE = 3
OUTPUT_SIZE = 2
NUM_COSYNE_CONNECTIONS = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE
SUBPOP_SIZE = 200
COSYNE_PARENTS = math.floor(0.25 * SUBPOP_SIZE)
COSYNE_MUTATION_RATE = 0.3
COSYNE_MUTATION_POWER = 0.3
WEIGHT_MAGNITUDE = 1


def solve_pole_cosyne(random_starts: bool, print_log: bool) -> (list[int], int):
    weight_population = init_population_COSYNE(NUM_COSYNE_CONNECTIONS, SUBPOP_SIZE)
    num_generation = 0
    evaluations = 0
    fitness_values = []

    while num_generation < 1000:
        network_population = weights_to_networks_pop(weight_population, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        best_fitness, evaluations = evaluate_population_cosyne(network_population, evaluations, random_starts)
        fitness_values.append(best_fitness)

        if best_fitness == MAX_STEPS:
            print(f"COSYNE found solution in generation {num_generation} after {evaluations} evaluations")
            break

        if print_log:
            print(f"Generation {num_generation}")
            print(f"Best fitness {best_fitness}")

        network_population = evolve_cosyne(network_population)
        weight_population = networks_to_weights_pop(network_population)
        weight_population = permute(weight_population)
        num_generation += 1

    return fitness_values, evaluations
```


```python
fitness_values, _ = solve_pole_cosyne(True, False)
x_range = np.arange(0, len(fitness_values), 1)

plt.plot(x_range, fitness_values)
plt.yscale("log")
```

    COSYNE found solution in generation 4 after 968 evaluations



    
![png](13/output_93_1.png)
    


### Evaluation of CoSyNE

We evaluate CoSyNE by comparing it against SANE in the static and randomised pole-balancing task using similar hyperparameter.


```python
def evaluate_NE_algorithms(random_start: bool, repetitions: int) -> (int, int, int):
    fitness_SANE = []
    fitness_COSYNE = []
    evaluations_SANE = []
    evaluations_COSYNE = []
    for _ in range(repetitions):
        fitness, evaluations = solve_pole_sane(random_start, False)
        fitness_SANE.append(fitness)
        evaluations_SANE.append(evaluations)

        fitness, evaluations = solve_pole_cosyne(random_start, False)
        fitness_COSYNE.append(fitness)
        evaluations_COSYNE.append(evaluations)

    return fitness_SANE, evaluations_SANE, fitness_COSYNE, evaluations_COSYNE


def plot_NE_algorithms(fitness_SANE: list[list[int]], fitness_COSYNE: list[list[int]]):
    # Padding
    plt.yscale("log")
    max_size = max([max([len(f) for f in f_list]) for f_list in [fitness_SANE, fitness_COSYNE]])
    for fitness_list in [fitness_SANE, fitness_COSYNE]:
        for f in fitness_list:
            while len(f) < max_size:
                f.append(MAX_STEPS)

    labels = ["SANE", "COSYNE"]
    colours = ["darkorange", "royalblue"]
    style = ['solid', 'dotted']
    for n, fitness_list in enumerate([fitness_SANE, fitness_COSYNE]):
        mean_fitness_time = np.array(fitness_list).mean(axis=0)
        x_range = np.arange(0, max_size, 1)
        plt.plot(x_range, mean_fitness_time, color=colours[n], label=labels[n], linestyle=style[n])

    plt.legend()
```


```python
repetitions = 10
random_start = True
f_SANE, eval_SANE, f_COSYNE, eval_COSYNE = evaluate_NE_algorithms(random_start, repetitions)
plot_NE_algorithms(f_SANE, f_COSYNE)
print(f"Average number of evaluations SANE: {mean(eval_SANE)}\nAverage number of evaluations COSYNE: {mean(eval_COSYNE)}")
```

    SANE found solution in generation 84 after 16870 evaluations
    COSYNE found solution in generation 14 after 2844 evaluations
    SANE found solution in generation 10 after 2107 evaluations
    COSYNE found solution in generation 0 after 55 evaluations
    SANE found solution in generation 0 after 114 evaluations
    COSYNE found solution in generation 3 after 643 evaluations
    SANE found solution in generation 13 after 2663 evaluations
    COSYNE found solution in generation 4 after 866 evaluations
    SANE found solution in generation 48 after 9616 evaluations
    COSYNE found solution in generation 21 after 4218 evaluations
    SANE found solution in generation 34 after 6845 evaluations
    COSYNE found solution in generation 11 after 2329 evaluations
    SANE found solution in generation 175 after 35069 evaluations
    COSYNE found solution in generation 0 after 118 evaluations
    SANE found solution in generation 57 after 11442 evaluations
    COSYNE found solution in generation 34 after 6801 evaluations
    SANE found solution in generation 50 after 10050 evaluations
    COSYNE found solution in generation 10 after 2087 evaluations
    SANE found solution in generation 0 after 107 evaluations
    COSYNE found solution in generation 7 after 1588 evaluations
    Average number of evaluations SANE: 9488.3
    Average number of evaluations COSYNE: 2154.9



    
![png](13/output_96_1.png)
    

