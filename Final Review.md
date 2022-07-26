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

So data modelling

<img src="https://i.postimg.cc/kgg41Fzm/Snipaste-2022-05-10-18-11-21.png" alt="2" style="zoom:50%;" />

- predictive models: 

  **Inferential Analysis (predictive)** is the type of analysis that can describe measures over the population of data. **That is observed and unobserved.** Fuzzy inference systems use human reasoning to allow for the prediction to happen.

  Regression: Map inputs to an output ∈ R; 

  Classification: Map inputs to a categorical output

- descriptive models: 

  **Descriptive Analysis** is the type of analysis and measure that seek to **describe and summarize the data** and the available samples. We **can not in general use it for the interpretation of unobserved data.**

we aim to find the function. 

## Linear Regression

from cov to correlation- 2 definitions

cor: only the measure of a **linear** relation. 

cor near 0: non-linear relation but maybe have another relation (maybe square...)



## ANN

Artificial Neural Networks (ANNs) are physically cellular systems, which can acquire, store and utilize experiential knowledge.

### Perceptron

#### Activation function

continuous, differentiable. There is one example:
$$
sigmoid(x)=\frac{1}{1+e^{-x}}
$$

#### Architecture

![1](https://i.postimg.cc/hvPYPcR7/1.jpg)

#### Training Algorithm

Input signals: x1,x2,...,xl. Adjustable weights: w1, w2, ..., wl and bias θ.

the line is $w_1x_1+w_2x_2-\theta=0$
$$
f(x)=sign(wx+\theta)
$$

1. Initialize weights and thresholds to small random values

2. Choose an input-output pattern$ (x^{(k)},t^{(k)})$ from the training data.

3. Compute the network’s actual output $c^{(k)}=f(\sum^I_{i=1}w_ix^{(k)}_i+\theta)$

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

The process of iterative optimization of an algorithm must be completed in a limited number of steps. The proof of the coverage of perceptron algorithm: A1q1

### Multi-Layer Perceptrons (MLPs)

![2](https://i.postimg.cc/t4np5kMT/2.jpg)

计算卷积sizehttps://zhuanlan.zhihu.com/p/205453986
