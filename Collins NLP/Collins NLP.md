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
- A **Language Model** is then $\mathcal{V}$ and a probability distribution $p(x_1,x_2,\dots,x_n)$:
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

