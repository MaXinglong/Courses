[TOC]



# Language Modeling

## Introduction

- Goal: construct a *language model* from a set of example sentences.
- Assume that we have a *corpus.* Let $\mathcal{V}$ be the set of all words in the language.
- A *sentence* in the language is a sequence of words

$$
x_1 x_2 \dots x_n
$$

where $n\geq 1$, $x_i \in \mathcal{V}$ for $1 \leq i \leq n-1$, and $x_n = STOP$ is a special symbol (that is not a member of $\mathcal{V}$).

- $\mathcal{V}^{\dagger}$ is the set of all sentences with the vocabulary $\mathcal{V}$.
- A **Language Model** is $\mathcal{V}$ and a probability distribution $p(x_1,x_2,\dots,x_n)$:
  - For any $\langle x_1 \dots x_n \rangle \in \mathcal{V}^{\dagger}, p(x_1,\dots,x_n)\geq 0$
  - $\sum_{\langle x_1 \dots x_n \rangle \in \mathcal{V}^{\dagger}} p(x_1,\dots,x_n)=1$
- Example (a very bad one): define $c(x_1, \dots, x_n)$ to be the number of times the sentence $x_1, \dots, x_n$ appears in the training corpus. We can define

$$
p(x_1 \dots x_n) = \frac{c(x_1 \dots x_n)}{N}
$$

where $N$ is the total number of sentences in the training corpus. This is a bad model because we assign probability $0$ to any sentence that does not appear in the training corpus.

## Markov Models

### Markov Models for Fixed-length Sequences

- Consider a sequence of random variables $X_1, X_2, \dots, X_n$, each taking a value from $\mathcal{V}$.
- For now, assume that $n$ is fixed. Later we will deal with the cause where $n$ is itself random.
- Our goal is to model

$$
P(X_1 = x_1, X_2 = x_2, \dots, X_n = x_n)
$$

- There are $|\mathcal{V}|^n$ possible sequences, so we can't just list all of the probabilities. We need a more compact model.
- To simplify the model, we make the following assumption (**first-order Markov process**):

$$
\begin{align}
P(X_1 = x_1, \dots, X_n = x_n) &=P(X_1 = x_1) \prod_{i=2}^n P(X_i = x_i | X_1 = x_1, \dots X_{i-1} = x_{i-1}) \\
&= P(X_1 = x_1) \prod_{i=2}^n P(X_i = x_i | X_{i-1} = x_{i-1})
\end{align}
$$

The first step is exact (chain rule). The second step makes the first order Markov assumption assumption:
$$
P(X_i=x_i | X_1=x_1 \dots x_{i-1}=x_{i-1}) = P(X_i = x_i | X_{i-1}=x_{i-1})
$$
(The identity of the ith word in the sequence depends only on the identity of the previous word)

- **Second-order Markov assumption:**

$$
\begin{align}
P(X_1 = x_1, \dots, X_n = x_n) &=P(X_1 = x_1) \prod_{i=2}^n P(X_i = x_i | X_1 = x_1, \dots X_{i-1} = x_{i-1}) \\
&= P(X_1 = x_1) \prod_{i=2}^n P(X_i = x_i | X_{i-2}=x_{i-2}, X_{i-1} = x_{i-1})
\end{align}
$$

For convenience, we will assume that $x_0 = x_{-1} = *$, where $*$ is a special "start" symbol.

### Markov Sequences for Variable-length Sentences

To generate a sequence of varying length:

1. Initialize $i=1$, and $x_0=x_{-1}=*$
2. Generate $x_i$ from the distribution

$$
P(X_i=x_i| X_{i-2}=x_{i-2}, X_{i-1}=x_{i-1})
$$

3. If $x_i=STOP$ then return the sequence $x_1 \dots x_i$. Otherwise, set $i=i+1$ and return to step 2.

## Trigram Language Models

- As in Markov models, we model each sentence as a sequence of $n$ random variables. The length itself is also a random variable, and we always have $X_n=STOP$. By the second-order Markov assumption, the probability of any sentence is

$$
P(X_1=x_1 \dots X_n=x_n) = \prod_{i=1}^n P(X_i=x_i | X_{i-2}=x_{i-2}, X_{i-1}=x_{i-1})
$$

where as before $x_0=x_{-1}=*$.

- We parametrize the model by

$$
P(X_i=x_i|X_{i-2}=x_{i-2}, X_{i-1}=x_{i-1}) = q(x_i|x_{i-2},x_{i-1})
$$

or, in short, $q(w|u,v)$ for any $(u,v,w)$. Note that $w$ can be equal to $STOP$ and $u,v$ can be equal to $*$.

- The **trigram language model** is then given by:

$$
p(x_1 \dots x_n) = \prod_{i=1}^n q(x_i | x_{i-2}, x_{i-1})
$$

- Example:

$$
\text{the dog barks STOP}
$$

$$
p(\text{the dog barks STOP}) = q(the|*, *)\times q(dog|*, the) \times q(barks | the, dog) \times q(STOP |dog, barks)
$$

- The parameters must satisfy that for any trigram $u,v,w$:

$$
q(w|u,v)\geq 0
$$

and for any bigram $u,v,$
$$
\sum_{w \in \mathcal{V} \cup \{STOP\}} q(w|u,v)=1
$$

- There are around $|\mathcal{V}|^3$ parameters in the model, which is a very large number.

### Maximum-Likelihood Estimates

- Define $c(u,v,w)$ to be the number of times the trigram $(u,v,w)$ is seen in the data.
- Define $c(u,v)$ to be the number of times the bigram $(u,v)$ is seen in the data.
- For any $(w,u,v)$, we define

$$
q(w|u,v)=\frac{c(u,v,w)}{c(u,v)}
$$

- This way of estimating the parameters has some very serious problems. Recall that we have a very large number of parameters in our model. Because of this, many of our counts will be zero. Also, when the denominator is zero the estimate is not well defined.

### Evaluating Language Models: Perplexity

- How do we measure the quality of a language model?
- One way is using *perplexity*.
- Let $x^{(1)},x^{(2)},\dots,x^{(m)}$ be some test data sentences. Each sentence $x^{(i)}$ is a sequence of words $x_1^{(i)},\dots,x_{n_i}^{(i)}$.
- As before, assume each sentence ends with STOP.
- The test sentences are "held out".
- For any test sentence, we can measure its probability $p(x^{(i)})$ under the language model.
- We can then measure the quality of the model by the likelihood that it gives the test set:

$$
\prod_{i=1}^m p(x^{(i)})
$$

- The idea is that the higher the likelihood, the better the language model is at modeling unseen sentences.
- Let $M=\sum_{i=1}^m n_i$.
- The average log probability under the model is defined as:

$$
l=\frac{1}{M}\log_2\prod_{i=1}^Mp(x^{(i)})=\frac{1}{M}\sum_{i=1}^m\log_2p(x^{(i)})
$$

- The perplexity is then defined as

$$
2^{-l}
$$

- The smaller the perplexity, the better the model.
- What's the intuition behind this measure? Suppose that we have a dumb model that always predicts $q(w|u,v)=\frac{1}{N}$. Then

$$
\begin{align}
-l=\frac{1}{M}\log_2 \frac{1}{N}^M-l &= \frac{1}{M}\log_2 \frac{1}{N}^M \\
&= log_2\frac{1}{N}
\end{align}
$$

which means that the perplexity is equal to $N$. So with the dumb model the perplexity is equal to the vocabulary size. Perplexity can be thought of as the *effective* vocabulary size.

- Note that if some $q(w|u,v)$ is equal to zero then the perplexity blows up.

