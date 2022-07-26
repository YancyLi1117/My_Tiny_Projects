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

  Unsupervised[KSON, Hopfield]**No priori known desired output.**

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

![2](https://i.postimg.cc/t4np5kMT/2.jpg)

计算卷积sizehttps://zhuanlan.zhihu.com/p/205453986
