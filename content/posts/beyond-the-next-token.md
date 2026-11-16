---
title: "Beyond the Next Token: A Practical Guide to How LLMs Actually Work"
date: "2026-11-16"
topic: "AI"
tags: ["LLM", "Pre-training", "RLVR", "Transformers"]
summary: "\"It predicts the next word\" is true but incomplete. A walk through pre-training, weights and architecture, and post-training with RLVR, and why the next-token label no longer tells the whole story."
---

> This article synthesizes the evolution of Large Language Model (LLM) capabilities, from raw data ingestion to sophisticated reward-based reasoning. It is a reference guide for understanding the "black box" of modern AI.

## Introduction

The most common description of an LLM is: _"It predicts the next word."_

Technically, this is true. But it is also like saying a car's purpose is "to move wheels." It misses the engine, the steering, and the navigation system.

As of 2026, the "next-token predictor" model is a **zeroth-order approximation**: a useful starting point, but incomplete. Modern LLMs are not just mimicking text; they are exploring, reasoning, and optimizing for rewards.

This article breaks down the three stages of an LLM's life: **pre-training**, **weights & architecture**, and **post-training (RLVR)**, and explains why the "next-token" label no longer tells the whole story.

## Part 1: Pre-training, learning language by imitation

Pre-training is where the model learns the structure of language, facts, and basic reasoning patterns. It is a process of massive imitation.

### The core loop

Imagine a simple training loop. The model is fed a dataset of text (books, web pages, code). For every token in every sentence, it performs this operation:

1. **Context:** it reads the tokens so far (e.g. "The cat sat on the").
2. **Target:** it looks at the actual next token in the data (e.g. "mat").
3. **Update:** it adjusts its internal settings to make "mat" slightly more likely to appear next time.

**Key concept:** the model doesn't "understand" cats or mats. It learns that in this specific context, "mat" has appeared many times. It builds a statistical map of the world based on frequency.

### The dataset effect

If the training data contains:

- "The cat sat on the mat" (2 times)
- "The dog sat on the rug" (1 time)
- "The bird sat on the branch" (1 time)

The model learns that after "The cat sat on the," "mat" is the most probable answer. It doesn't know what a cat is; it just knows where the word "mat" belongs in the sequence.

**Summary:** pre-training teaches the model grammar, facts, and patterns by forcing it to predict what a human wrote next.

## Part 2: The engine, weights, biases, and architecture

After pre-training, the model is a frozen set of numbers and mathematical structures. To understand how it works, we need to look under the hood.

### 1. Weights (the knowledge)

- **What they are:** billions of floating-point numbers (e.g. `0.37`, `-1.94`).
- **Function:** each weight is a "knob" that controls how strongly one piece of information influences another.
- **Analogy:** if you have a dial board for every possible connection between ideas, the weights are the settings you turn to match reality.
- **Storage:** knowledge isn't stored in one place. It's distributed. The fact that "Paris is the capital of France" is encoded across millions of weights, not in a single database cell.

### 2. Biases (the thresholds)

- **What they are:** single numbers added to each neuron.
- **Function:** they act as a "baseline activation level." A high bias makes a neuron eager to fire; a low bias makes it skeptical.
- **Analogy:** if weights are the volume of a signal, bias is the sensitivity of the receiver.

### 3. Architecture (the blueprint)

- **What it is:** the fixed mathematical structure that defines how data flows through the model.
- **Key components:**
  - **Embeddings:** convert words into lists of numbers (vectors).
  - **Self-attention:** lets the model look at previous words and decide which ones matter most for the current prediction.
  - **Feed-forward networks:** layers where much of the "knowledge storage" happens.
  - **Residual connections & layer norm:** tricks that keep training stable.
- **Fixed vs. learned:** the architecture (number of layers, size of heads) is fixed before training. The weights and biases are the values adjusted during training.

**Analogy:** the architecture is the piano's design (keys, strings, hammers). The weights are the tuning of those strings. The architecture stays the same; the tuning changes.

## Part 3: Post-training, from imitation to reasoning

Pre-training gives the model knowledge. Post-training gives it behavior and reasoning ability. This is where the "next-token predictor" model falls apart.

### Phase A: Supervised fine-tuning (SFT)

- **Goal:** teach the model to be a helpful assistant.
- **Method:** humans write good Q&A pairs (e.g. "How do I boil an egg?" → "1. Fill pot...").
- **Result:** the model learns to respond politely and directly, rather than just completing sentences.

### Phase B: RLVR (Reinforcement Learning with Verifiable Rewards)

This is the modern standard for making LLMs "smart."

**How it works**

1. **Self-generated data:** the model is given a hard problem (e.g. a complex math proof). It generates its own answer.
2. **Reward model:** a separate checker (AI or code) grades the answer. Correct: reward +1. Wrong: reward -1.
3. **Optimization:** the model adjusts its weights to maximize the reward, not to mimic a human.

**The key difference**

- **Pre-training:** "What token did a human write next?" (imitation)
- **RLVR:** "Which token leads to a correct, winning outcome?" (optimization)

In RLVR, there is no "correct next token" from a dataset. The model explores different paths. If a weird, convoluted reasoning path leads to a correct answer, it gets rewarded. Next time, the model is more likely to take that path.

## Part 4: Why "next-token predictor" is the wrong mental model

Calling an LLM a "next-token predictor" is misleading because it describes the shape of the mechanism but ignores its purpose.

| Feature         | Pre-training (imitation)         | RLVR (optimization)                           |
| --------------- | -------------------------------- | --------------------------------------------- |
| Source of truth | Human-written data               | Reward signals                                |
| Goal            | Minimize surprise (fit the data) | Maximize reward (achieve the goal)            |
| Behavior        | Mimics patterns                  | Explores and reasons                          |
| Analogy         | A student copying a textbook     | A chess engine playing itself to find wins    |

### The chess analogy

- **System A (pre-training):** predicts the move a grandmaster would play. A next-move predictor.
- **System B (RLVR):** explores millions of games to find the move that leads to a win. A reward maximizer.

Calling System B a "next-move predictor" is strange. It doesn't care whether the move looks like something a human played. It cares whether it wins.

## Conclusion: what are we actually building?

Modern LLMs are not just text generators. They are:

- **Compressed knowledge bases:** billions of weights encoding statistical relationships from the internet.
- **Reasoning engines:** systems that use self-generated data and reward signals to discover strategies not present in the original training data.
- **Optimizers:** tools designed to maximize specific outcomes (helpfulness, correctness, safety) rather than just mimic language.

The "next-token" loop is still the interface we use to interact with them. But inside, they are doing something far more complex: simulating thought, exploring possibilities, and learning from their own mistakes.

## Glossary

- **Token:** a chunk of text (a word or part of a word) that the model processes.
- **Inference:** running the model to generate a response.
- **Pre-training:** the initial phase where the model learns from massive datasets of human text.
- **Post-training:** the phase where the model is fine-tuned for behavior and reasoning (SFT + RLVR).
- **RLVR:** Reinforcement Learning with Verifiable Rewards. The model learns by generating its own solutions and being graded on their correctness.
- **Weights:** the numerical parameters that store the model's "knowledge."
- **Bias:** a threshold value that helps neurons decide when to activate.
- **Attention:** the mechanism that lets the model focus on relevant parts of the input sequence.
- **Reward model:** a separate system that evaluates the quality of the LLM's output.

## References

- [LLM next-token predictors](https://gmcgoldr.github.io/2026/09/04/llm-next-token-predictors.html), the article this guide draws on.
