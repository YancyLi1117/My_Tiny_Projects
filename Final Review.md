# Final Review

> Yancy Li
>
> y3372li@uwaterloo.ca
>
> 2022.07.25

## Some basic concept

Artificial Intelligence: Algorithms enabled by constraints, exposed by representations that support models, and targeted at thinking, perception, and action.

Intelligent systems: artificial entities involving a mix of **software and hardware** which have a capacity to acquire and apply knowledge in an ”intelligent” manner and have the capabilities of perception, reasoning, learning, and making inferences (or, decisions) from incomplete information. **Intelligent systems use AI and ML.**

Feature of the intelligent system: the generation of **outputs**, based on some **inputs** and the nature of the system itself. 

Capability: sensory perception, pattern recognition...

Intelligent machine: a machine that can exhibit one or more intelligent characteristics of a human.  **An intelligent machine embodies machine intelligence**. (the physical elements of a machine are not intelligent but the machine can be programmed to behave in an intelligent manner.)

Difference between AI,ML,DL:

<img src="https://i.postimg.cc/pXBqvRTH/Snipaste-2022-05-10-18-02-10.png" alt="diff" style="zoom:50%;" />

So data modelling:

<img src="https://i.postimg.cc/kgg41Fzm/Snipaste-2022-05-10-18-11-21.png" alt="2" style="zoom:50%;" />

- predictive models: 

  **Inferential Analysis (predictive)** is the type of analysis that can describe measures over the population of data. **That is observed and unobserved.** Fuzzy inference systems use human reasoning to allow for the prediction to happen.

  Regression: Map inputs to an output ∈ R; 

  Classification: Map inputs to a categorical output

- descriptive models: 

  **Descriptive Analysis** is the type of analysis and measure that seek to **describe and summarize the data** and the available samples. We **can not in general use it for the interpretation of unobserved data.**

Pattern recognition: the assignment of some sort of output value (or label) to a given input value (or instance), according to some specific algorithm.

Classification: the problem of **identifying to which of a set of categories a new observation belongs**, on the basis of a training set of data containing observations whose category membership is known.

## Linear Regression

### covariance and correlation

$$
Cov=\frac{1}{n-1}\sum_{i=1}^n(X_i-\overline{X})(Y_i-\overline{Y})\\Cor(X,Y)=\frac{Cov(X,Y)}{S_X,S_Y}, -1\leq Cor(X,Y)\leq 1
$$

(Why cov denominator is n-1? [cov degree of freedom is n-1](https://www.zhihu.com/question/326157416/answer/698480209), [sample variance df is n-1](https://www.zhihu.com/question/20099757/answer/26586088))

cor: only the measure of a **linear** relation. 

cor near 0: **non-linear relation but maybe have another relation** (maybe square...)

### Maximum Likelihood Estimate (MLE)

$$
Y=Xw+\epsilon\quad \epsilon\sim N(0,\sigma^2I)\quad Y\sim N(Xw,\sigma^2)\\P(Y|Xw,\sigma^2I)=\frac{1}{(\sqrt{2\pi}\sigma^2)^n}\exp\{-\frac{1}{2\sigma^2}(Y-Xw)^{-1}(Y-Xw)\}
$$

$$
w_{ML}=arg\max_w\ln P(Y|\mu=Xw,\sigma^2I)\\=arg\max_w-\frac{1}{2\sigma^2}||Y-Xw||^2-\frac{n}{2}\ln (2\pi\sigma^2)\\=arg\min_w||Y-Xw||^2
$$

$$
L=||Y-Xw||^2=(y-Xw)^T(y-Xw)\\
\nabla_wL=2X^TXw-2X^TY=0\Rightarrow w_{LS}=(X^TX)^{-1}X^TY
$$

If $(X^TX)^{-1}$is not full rank, doesn't have an Analytic expression. Use gradient descent.

## Logistic Regression

**A form of regression** allows the prediction of **discrete** variables by a mix of continuous and **discrete** predictors.

Binary logistic regression is **a type of regression** analysis where the dependent variable is a dummy variable: coded 1-positive and 0 negative.

Output: Categorical or binary /  Probabilistic output.

### Sigmoid function

$\theta(x)=\frac{e^x}{1+e^x}$  But why this function? consider log odds. 
$$
L=\ln \frac{p(y=+1|x)}{p(y=-1|x)},\quad p(y=+1|x)+p(y=-1|x)=0
$$
When L>>0, trust y=+1; L<<0, trust y=-1; L=0, either

### MLE in logistic regression

Connect with Linear Function,

The posterior probability:
$$
P(y_i=+1|x_i,w)=\theta(x_i^Tw)=\frac{e^{x_i^Tw}}{1+e^{x_i^Tw}}\\P(y_i=-1|x_i,w)=1-\theta(x_i^Tw)=\theta(-x_i^Tw)
$$
From (6), we have: $P(y_i|x_i,w)=\theta(y_ix_i^Tw)$.

Then the composite probability:（independent events... so product the probability)
$$
p(y1,...y_n|x_1,...x_n,w)=\prod_{i=1}^np(y_i|x_i,w)\\=\prod_{i=1}^n\theta(y_ix_i^Tw)
$$
Given: $\theta(x)=\frac{e^x}{1+e^x}$ , so $\frac{1}{\theta(x)}=1+e^{-x}, \theta(x)'=\theta(x)(1-\theta(x))$, 
$$
w_{ML}=\arg\max_w\sum_{i=1}^n\ln \theta(y_ix_i^Tw)=\arg\max_w L\\
\nabla_wL=\sum_{i=0}^n(1-\theta(y_ix_i^Tw))y_ix_i^T
$$
Use Newton- Raphson method to look for w

## ANN

Artificial Neural Networks (ANNs) are physically cellular systems, which can acquire, store and utilize experiential knowledge.

Characteristic:

- Architecture

  Feedforward [**output won't give to input**]

  Recurrent

- Learning paradigm

   Supervised[MLP, RBFN] Particularly useful for feedforward networks. **priori known desired outpu**t

  Unsupervised[KSON, HNN]**No priori known desired output.**

  Reinforcement Learning: Network’s connection **weights are adjusted** according to a qualitative and not quantitative feedback information

  **Training algorithm and pictures** (set2 11 14 17)

- Activation functions

  continuous, differentiable. There is one example: $sigmoid(x)=\frac{1}{1+e^{-x}}$

### Perceptron

Mainly designed to **classify linearly separable** (exists a hyperplanar multidimensional decision boundary that classifies the patterns into two classes.) patterns, **supervised learning** 

#### Perceptron Convergence Theorem

There exists a set of weights for which the training of the Perceptron will **converge in a finite time** and the training patterns are correctly classified. The proof of the coverage of perceptron algorithm: A1q1

#### Architecture

<img src="https://i.postimg.cc/hvPYPcR7/1.jpg" alt="1" style="zoom:50%;" />

#### Training Algorithm

Input signals: x1,x2,...,xl. Adjustable weights: w1, w2, ..., wl and bias θ.

the line is $w_1x_1+w_2x_2-\theta=0$
$$
f(x)=sign(wx+\theta)
$$

1. Initialize weights and thresholds to small random values

2. Choose an input-output pattern$ (x^{(k)},t^{(k)})$ from the training data.

3. Compute the network’s actual output $o^{(k)}=f(\sum^I_{i=1}w_ix^{(k)}_i+\theta)$

4. Adjust the weights and bias according to the perceptron learning rule: $∆w_i = η[t^{(k)} − o^{(k)}]$, and $∆θ = −η[t^{(k)} − o^{(k)}]$. where $η ∈ [0, 1]$ is the perceptron’s learning rule. If f is the signum function, this becomes equivalent to
   $$
   \Delta w_i=\left\{
   \begin{aligned}
   2\eta t^{(k)}x_i^{(k)}\quad &if\ t^{(k)}\neq o^{(k)}\\
   0\quad & otherwise  \\
   \end{aligned}
   \right.
   \\
   \Delta \theta=\left\{
   \begin{aligned}
   -2\eta t^{(k)}\quad &if\ t^{(k)}\neq o^{(k)}\\
   0\quad & otherwise  \\
   \end{aligned}
   \right.
   $$
   
5. If a whole epoch is complete, then pass to the following step; otherwise, go to Step 2. 

6. If the weights (and bias) reached a steady-state  $(∆wi ≈ 0)$ through the whole epoch, then stop the learning; otherwise go through one more epoch starting from Step 2.

**The example on the slide!**

### Multi-Layer Perceptrons (MLPs)

The perceptron **lacks** the important capability of **recognizing patterns belonging to non-separable linear spaces**.

MLP solves a wide range of complex problems.

<img src="https://i.postimg.cc/t4np5kMT/2.jpg" alt="2" style="zoom:67%;" />

#### Backpropagation Learning Algorithm

**Online learning** (also called *incremental learning*): The weight changes made at a given stage depend specifically **only** on the (current) example being presented and possibly on the current state of the model. It is the natural procedure for **time-varying** rules where the examples might not be available at all at once.

**Offline learning**: the weight changes **depending on the whole (training) dataset**, defining a global cost function. The examples are **used repeatedly** until minimization of this cost function is achieved.

the derivation: https://www.cnblogs.com/jsfantasy/p/12177275.html

https://blog.csdn.net/z_feng12489/article/details/89187037

#### Learning Algorithm

P19 

#### Momentum

learning parameter η small, very slow convergence rate of the algorithm; large, lead to unwanted oscillations in the weight space.
$$
∆w(l)(t+1)=−η\frac{∂E_c(t)}{\partial w^l} +γ∆w^l(t)
$$
example of MLP on slides

more samples, a proper number of hidden layers ---- better output.

#### **MLP application**

Signal processing, Weather forecasting, Pattern recognition, Signal compression, Financial market prediction

#### Limitation

The gradient descent-based algorithm used to update the network weights may **never converge to the global minima**, particularly in the case of highly nonlinear behaviour of the system being approximated by the network

Using optimization techniques such as those based on: Genetic algorithms, Simulated annealing.

### Radial Basis Function Network

**----feedforward neural networks architecture**.

an input layer, a single hidden layer with radial activation function and an output layer.

#### Topology

<img src="https://i.postimg.cc/4xbghFK8/657s31.jpg" style="zoom:50%;" />

hidden layer; **nonlinear** transformations, **symmetrical** (typical transfer functions for hidden functions are Gaussian curves).

between the hidden and output layers: **linear** transformations

weights between the input layer and hidden layer: equal to unity.

（input spaces, cast nonlinearly into high-dimensional domains, are more likely to be linearly separable than those cast into low-dimensional ones.）

RBF function: x: input, $v_i$ the center of radial function, $\sigma_i$ width parameter
$$
g_i(x)=r_i\frac{||x-v_i||}{\sigma_i}
$$
Gaussian kernel function and logistic function are possible RBF function
$$
o_i(x)=\sum_{i=1}^nw_{ij}g_i(x),j=1,...,r
$$

#### Learning Algorithm

Step 1: Train the RBF layer to **get the adaptation of centers** and scaling parameters using **unsupervised training.**

use clustering: K-means, MLE, Self-organizing map method.

Step 2: Adapt the weights of the output layer using **supervised training.**

using Least-squares method, Gradient method to update the weight, use the inverse or pseudo-inverse method to calculate the weight matrix.

Here use Gaussian kernel as the radial basis function, D = GW, where D is the desired output of the training data.

If $ G^{-1}$ exists, $W= G^{-1}D$. 

But when G is ill-conditioned use: $W=G^+D$, where $G^+=(G^TG)^{-1}G^T$

**Choose a proper width**

Too small: can't provide a good interpolation between sample data. Too large:  lose a lot of information when the ranges of the radial functions are further away from the original range of the function.

#### Advantages/Disadvantages

pro: train **faster** than a MLP, the hidden layer is **easier to interpret** than the hidden layer in an MLP.

Con: training is finished and it is being **used it is slower** than a MLP, **unsupervised learning stage is not an easy task**,have an undesirably **high number of hidden nodes** (but the dimension of the space can be reduced by careful planning of the network.)

#### Application

have universal approximation capabilities, good local structures and efficient training algorithms

- Nonlinear mapping of complex processes and for solving a wide range of classification problems.
- control systems, audio and video signals processing, and pattern recognition.
- chaotic time series prediction, with particular application to weather and power load forecasting.

### Kohonen’s Self-Organizing Network

**----unsupervised learning networks**

#### Topology

Competitive learning: the nodes distribute themselves across the input space to **recognize groups of similar input** vectors, two input vectors with similar pattern characteristics excite **two physically close layer nodes**. The **output nodes compete among themselves** to be fired one at a time in response to a particular input vector.

<img src="https://i.postimg.cc/6qZpX4kP/657s32.jpg" style="zoom:50%;" />

#### Learning Algorithm

slide p88, and example

$N_c$: neighbourhood around the winning output candidate. decreases at every iteration until convergence occurs.

Step1: Initial all weights, learning rate, $N_c$, choose pattern x from the input

Step2: Select winning unit c, let $I=||x-w_c||=\min_{ij}||x-w_{ij}||$ minimized

Step3: Update the weights, and the learning rate and the neighbourhood are decreased at every iteration

Step4: Continue until each output reaches a threshold of sensitivity to a portion of the input space.

Update the weight of neighbour: Closer neighbours are rewarded more than the ones that are farther slide p107

#### Application

**Clustering applications** such as Speech recognition, Vector coding, Robotics applications, and Texture segmentation.

### Hopfield Network

**----recurrent topology**

#### Topology

Associative Memory Concept: 

- **recognize newly presented patterns using an already stored** ’complete’ version of that pattern.
- energy function that keeps decreasing until the system has reached stable status.

**Processing units** configured in **one single layer** (besides the input and the output layers) with symmetrical synaptic connections

<img src="https://i.postimg.cc/dVYtGVyR/657s33.jpg" style="zoom:67%;" />
$$
o_i=sign(\sum_{j=1}^nw_{ij}o_j-\theta_i)
$$
Energy function: **keeps decreasing until reaches stable status**.
$$
E=-1/2\sum\sum_{i\not=j}w_{ij}o_io_j+\sum o_i\theta_i\\\Delta E=-1/2\Delta o_i\sum_{i\not=j}w_{ij}o_j-\theta_i
$$

#### Hebbian Learning

When two units are simultaneously activated, their interconnection weight increase becomes proportional to the product of their two activities. https://zh.wikipedia.org/wiki/%E8%B5%AB%E5%B8%83%E7%90%86%E8%AE%BA
$$
w_{ij}=\left\{
\begin{array}
1/n\sum_{k=1}^qp_{kj}p_{ki}&&{i\not=j}\\
0&&{i=j}\\
\end{array} \right.
$$
slide p125, and example

HNN has the ability to remember the fundamental memory and its complement.

#### Application

- Information retrieval and for pattern and speech recognition, 
- Optimization problems,
- Combinatorial optimization problems such as the traveling salesman problem.

Limitation:

Limited stable-state storage capacity of the network
Many studies have been carried out recently to increase the capacity of the network without increasing much the number of the processing units

## Introduction to Deep Learning

Why Do We Need Deep Connectionist Models?

- Shallow models require a larger number of units, difficult to train (over-fitting is a major issue).
- Deep models lead to hierarchical representations, with varying levels of abstraction.

Why Weren’t Deep Models Popular Before? ----difficult to train

Gradient flow problems--Many heuristics and schemes
Need large datasets--immediate access to large datasets
Need large computer and memory resources--easy access to powerful computing power and open-source software libraries

### Convolutional Neural Networks

Contain at least 1 convolutional layer, Pooling layers, Fully-connected layers, Element-wise activation layers

<img src="https://i.postimg.cc/sgzhM8PK/657s43.jpg" style="zoom:50%;" />

#### Convolutional layers

differ from fully-connected layers in 2 ways:

- **Local connectivity**:  connect only to part of the input 
- **Parameter sharing**:  has the same parameters and represents a single convolutional unit.

Convolution Operator: $z(t)=(w\star x)(t)=\sum_{-\infty}^\infty x(t+\tau)w(\tau)$

Convolutional units go by many names: filters, kernels, and conv units. Hyperparameters: Filter size, Number of filters in a layer, Stride, Padding. Calculate the size of outputs: https://zhuanlan.zhihu.com/p/205453986

CNNs can be 1D, 2D, or 3D, depending on the type of input. 

#### Pooling Layers

Added after convolution layers, to **reduce the size** of the feature maps outputted by convolution layers, include max pooling and average pooling

Pooling layers don’t have learned parameters, build needs hyperparameters. Pooling size (larger pooling size, smaller output shape), Stride (Larger strides, smaller shapes)

Pros: Add some translation invariance to CNNs,  allow some tolerance, edge can appear in the feature map without affecting the classification result.

Cons: output space has a lower dimension, information is lost; not fit within the ‘end to end’ learning paradigm of deep learning.

Application: AlexNet, LeNet, VGG, GoogleLeNet, ResNet

### Recurrent Neural Networks

Assume some sort of **dependency between data** samples.

<img src="https://i.postimg.cc/kGtnsJC8/657s42.jpg" style="zoom:40%;" />
$$
h_y=\sigma_h(W_{xh}\centerdot x_t+W_{hh}\centerdot h_{t-1}+b_h)\\o_t=\sigma_o(W_{ho}\centerdot h_t+b_o)
$$
$\sigma$: activation functions; b: bias, learning the parameters W and b using backpropagation through time (BPTT).

Due to vanishing and exploding gradient, many variants of RNNs have developed, such as LSTM

### Autoencoders

**---unsupervised models**

Map the input onto itself with the restriction that one of the hidden layers(bottleneck), has a **lower dimension than the input**.

Used for **dimensionality reduction**, learn **non-linear mappings** from input to encoding space.

<img src="https://i.postimg.cc/C15MZC6c/657s41.jpg" style="zoom:40%;" />

### Natural Language Processing

#### Application

Detect new words, Language learning, Machine translation, NL interface, Information retrieval

#### Word Representation

One-hot encoding, Word occurrence (Term frequency), Term frequency-Inverse Document Frequency(TF-IDF), Word embedding, Latent Semantic Analysis, Word2Vec(a software package for representing words as vectors, containing two distinct models:CBoW, Skip-Gram), GloVe

Weaknesses of Word Embedding: Very vulnerable, and not a robust concept; Can take a long time to train; Non-uniform results;  Hard to understand and visualize

## **Introduction to Soft Computing and Intelligent Systems**

Knowledge-based system: **make perceptions** and new **inferences or decisions** using its reasoning mechanism

Need two types of knowledge: **knowledge of the problem**, knowledge regarding methods for **solving the problem**

Ways of representing and processing knowledge: Logic, Semantic Networks, Frames, Production Systems, and Fuzzy Logic.

Two types of logic: **crisp** (**binary**,  deals with statements called ”propositions”) and **fuzzy** (multivalued).

 In logic, knowledge is represented by **propositions**, connected by logical connectives such as AND, OR, NOT, EQUALS, IMPLIES.

The typical end objective of knowledge processing is to **make inferences**. 

The inference process used in a rule-based system is **deductive inference**  (Forward Chaining, Backward Chaining)

Rules of inference:

- Conjuction: $(A,B)\Rightarrow A\wedge B$
- Modus Ponens: $A\wedge(A\rightarrow B)\Rightarrow B$
- Modus Tollens: $\overline{B}\wedge(A\rightarrow B)\Rightarrow \overline{A}$
- Hypothetical Syllogism:$(A\rightarrow B)\wedge(B\rightarrow C)\Rightarrow A\rightarrow C$ 

Soft computing: an important **branch of intelligent and knowledge-based system**s, Fuzzy logic, probability theory, neural networks, and genetic algorithms are cooperatively used in soft computing

Fuzzy logic (FL) allows for a **realistic extension of binary, crisp logic to qualitative, subjective, and approximate situations**

GA: **derivative-free** optimization techniques, in the development of an optimal and **self-improving intelligent machine.**

<img src="https://i.postimg.cc/d0hCP57w/657s51.jpg" style="zoom:50%;" />

Characteristics: based on **multiple searching points or solution candidates**; use **evolutionary operations**; based on **probabilistic operations**

### Fuzzy logic 

Applications of Fuzzy Logic:

<img src="https://i.postimg.cc/pVsqH6rn/657s61.jpg" style="zoom:50%;" />

#### Fuzzy set

X: universe of discourse. represented by a membership function, values in the interval [0, 1].

**A crisp set is a special case of a fuzzy set** where the membership function can take only two values, 0 and 1.

<img src="https://i.postimg.cc/yYQfKrfb/657s62.jpg" style="zoom:40%;" />

universe of discourse: maybe discrete or continuous

#### Member functions

Triangular and trapezoidal membership functions:  Linear ⇒ Computationally inexpensive. Non-differentiable ⇒ May not be suitable to be used with gradient-descent optimization algorithms.

Gaussian and Bell membership functions: Differentiable ⇒ Suitable to be used with gradient-descent optimization algorithms. Nonlinear ⇒ Computationally expensive.

##### Triangular

3 parameters {a, b, c}, a < b < c.

<img src="https://i.postimg.cc/Ghbbs9xy/657s63.jpg" style="zoom:50%;" />

##### Trapezoidal

4 parameters {a, b, c, d}, a < b < c < d.

<img src="https://i.postimg.cc/9QVJGRxy/657s64.jpg" style="zoom:50%;" />

##### Gaussian

2 parameters {σ,c}

<img src="https://i.postimg.cc/7LJGhXg9/657s65.jpg" style="zoom:50%;" />

##### Generalized Bell

3 parameters {a, b, c}, a: the shape(smaller a, thinner), b: the top area(largerb, flatter), c: the location

<img src="https://i.postimg.cc/K8DFNTmb/657s66.jpg" style="zoom:50%;" />

#### Fuzzy Logic Operations

- Complement; μA′(x)=1−μA(x), x∈X
- Union: μ*A*∪*B*(x) = max(μ*A*(*x*),μ*B*(*x*)),∀*x* ∈ X
- Intersection: μ*A*∩*B*(x) = min(μ*A*(*x*),μ*B*(*x*)),∀*x* ∈ X

##### Generalized Fuzzy Complement

Boundary conditions, Non-increasing,Involutive

Look at the 457b note---
