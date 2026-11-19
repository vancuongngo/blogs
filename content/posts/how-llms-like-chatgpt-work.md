---
title: "How LLMs Like ChatGPT Work: A Plain-Words Guide"
date: "2026-11-17"
topic: "AI"
tags: ["LLM", "Tokenization", "Pre-training", "SFT", "RLHF", "Karpathy"]
summary: "Andrej Karpathy's 3.5-hour deep dive, rewritten as a readable guide: how data, tokens, pre-training, fine-tuning and reinforcement learning turn a next-token guesser into ChatGPT, with self-check questions."
---

An LLM like ChatGPT is a big math function that reads a list of text pieces (tokens) and guesses the next piece, one at a time. Everything else — knowledge, personality, "thinking" — comes from what data it was trained on and how.

This article rewrites Andrej Karpathy's 3.5-hour video [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) as a readable guide. The numbers and model names are as of early 2025, when the video was made.

A model is built in three stages, run one after the other, usually by separate teams:

1. **Pre-training** — read a huge slice of the internet and learn to predict the next token. Output: a _base model_, an "internet text simulator". This is the expensive part: months, thousands of GPUs, millions of dollars.
2. **Supervised fine-tuning (SFT)** — keep training, but on example conversations written by people. Output: an _assistant_ that answers questions. Takes hours, not months.
3. **Reinforcement learning (RL)** — let the model try many answers to practice problems and reinforce the ones that work. Output: a _thinking_ model that has found its own ways to reason.

A handy way to remember it: it is like a school textbook. Pre-training is reading the explanations. SFT is studying the worked examples. RL is doing the practice problems at the end of the chapter, where you only know the final answer.

| Stage | Data | Effort | Output |
| --- | --- | --- | --- |
| 1. Pre-training | ~15T tokens of web text | Months, thousands of GPUs | Base model |
| 2. Supervised fine-tuning | Example conversations | Hours, written by labelers | Assistant |
| 3. Reinforcement learning | Problems with answers | Model practices, keeps winners | Thinking model |

The same next-token training runs in every stage; only the data changes.

## Stage 1: Pre-training — download the internet

The first step is to collect a huge amount of good, varied text. Hugging Face's public FineWeb dataset is a typical example: after heavy filtering it is only about **44 TB** of text — it would almost fit on one hard drive. Big labs (OpenAI, Anthropic, Google) build their own version of the same thing.

Most of the raw data comes from **Common Crawl**, a group that has crawled the web since 2007 (2.7 billion pages indexed by 2024). That raw data is cleaned in several steps:

1. **URL filtering** — drop whole domains on block lists: malware, spam, marketing, adult, racist sites.
2. **Text extraction** — throw away HTML, CSS and menus; keep only the page's real text.
3. **Language filtering** — guess each page's language. FineWeb keeps a page only if a language classifier is more than 65% sure it is English (a score above 0.65). That is a confidence score, not the share of English words. This is a design choice: if you drop most Spanish, the model will be weak at Spanish.
4. **Deduplication and quality filters** — remove copies and junk.
5. **PII removal** — try to remove pages with addresses, social security numbers and similar personal data.

What is left is just plain text from millions of web pages, like an article about tornadoes in 2012 or a medical page about adrenal glands. Glue it all together and you get one gigantic stream of text. The goal of pre-training is to train a neural network to copy the patterns in how that text flows.

## Tokenization — turning text into symbols

A neural network needs its input as a one-dimensional list of symbols, picked from a fixed set (the _vocabulary_). There is a trade-off: fewer kinds of symbols means longer lists, and list length is a precious, expensive resource. So we want more symbols and shorter lists.

How text gets there, step by step:

| Representation | Number of possible symbols | Length of the list |
| --- | --- | --- |
| Raw bits (UTF-8) | 2 (0 and 1) | very long |
| Bytes (groups of 8 bits) | 256 | 8× shorter than bits |
| Tokens (after byte-pair encoding) | ~100,000 (GPT-4: 100,277) | much shorter again |

**Byte-pair encoding (BPE)** works like this: find the pair of symbols that appears most often next to each other (say byte 116 followed by byte 32), give that pair a new symbol (ID 256), and replace every occurrence. Repeat until the vocabulary is the size you want. Each round makes the list shorter and the vocabulary bigger.

You can play with this on the website Tiktokenizer (pick `cl100k_base`, the GPT-4 tokenizer). Some examples:

- "hello world" = 2 tokens: "hello" (ID 15339) and " world" (ID 1917). Note the space belongs to the second token.
- Two spaces between the words gives a different split, with an extra token 220.
- It is case sensitive: "Hello" is a different token from "hello".

Think of token IDs as names, not numbers — like 100,000 different emojis. FineWeb's 44 TB becomes a sequence of about **15 trillion tokens**. That sequence is what the model learns from.

## Training the neural network — learn to guess the next token

Training is a simple game repeated trillions of times: show the network some tokens, ask what comes next, and nudge it toward the right answer.

**One training step, concretely.** Take a random window of tokens from the data — in the video, 4 tokens: "|", " View", "ing", " Single". (Real windows are up to ~8,000 tokens; that limit is the _context length_.) The next token in the data is " Post" (ID 3962).

1. Feed the 4 tokens in. The network outputs **100,277 numbers** — one probability for every token in the vocabulary.
2. At the start, the network is random. In the video's illustration, " Post" gets 3%, " Direction" 4%, token 11799 2%. (A truly untrained model gives every token about 1/100,277 ≈ 0.001%.)
3. We know the right answer is " Post". A math procedure (gradient descent) adjusts the network so " Post" goes up a bit (say to 4%) and the others go down.
4. Do this for every token in large batches, in parallel, over and over.

Over time, the network's guesses match the statistics of how tokens follow each other in the data.

**What is inside.** The network is a big but fixed mathematical expression. It mixes the input tokens with **parameters** (also called weights): billions of numbers that start random. Think of them as knobs on a DJ mixer. Training just means finding a setting of the knobs that makes good predictions.

The modern design is called the **Transformer**. Tokens first become vectors (embeddings), then flow through layers of simple operations — matrix multiplications, "attention", small MLP blocks, softmax — until they come out as next-token probabilities. You can loosely call the numbers inside "neurons", but they are far simpler than brain cells: there is no memory, the same input always produces the same output.

**The loss.** The one number researchers watch is the _loss_: how wrong the predictions are on average. Lower is better. In Karpathy's GPT-2 reproduction:

- Each step improves predictions on **1 million tokens** at once and takes about **7 seconds**.
- The full run is **32,000 steps** ≈ **33 billion tokens**.
- After 20 steps the model writes random junk. At 1% of training it writes locally sensible but meaningless English. By the end it writes fairly coherent English.

**The hardware.** This runs on GPUs such as the Nvidia H100, which are good at doing many multiplications in parallel. Karpathy rented a machine with 8 H100s for about **$3 per GPU per hour**. Big labs put tens or hundreds of thousands of GPUs in one data center (for example, xAI's 100,000-GPU cluster). All of them are doing the same thing: predicting the next token on a dataset like FineWeb. This demand is why Nvidia was worth about $3.4 trillion at the time.

### Check yourself: data, tokens and one training step

Try to answer in your own words first. The answers are at the end of the article.

**Data**

1. FineWeb's classifier gives a page an English score of 0.60. Is it kept? Why can a page with 80% English words still be dropped?
2. A model is trained only on FineWeb. Why is it weak in Vietnamese? Name two ways to fix that.

**Tokens**

3. "hello hello" is 11 bytes. Merge the pair (l, l) into new ID 256. What are the new list length and vocabulary size?
4. Why not just use the 256 byte values as the vocabulary? What is the trade-off?
5. How many bytes are "t", "ñ" and 😀? Is every one of those bytes still a value from 0 to 255?

**One training step**

6. The text is "The cat sat on the mat" (6 tokens) and the context length is 4. Write one window's inputs and targets. How many guesses does that window give?
7. In your window, which tokens may the position at "cat" look at? What would go wrong without the mask?
8. A step uses 16 windows of length 4. How many guesses are made, and how many times are the parameters adjusted?
9. The model gives the correct next tokens these probabilities: 0.5, 0.25, 0.1, 1.0. Compute the average loss (−ln of each, then the mean).
10. Before any training, about what probability does each token get, with a vocabulary of 100,277?
11. A model has a context length of 1,024. You paste 3,000 tokens. Name two things the app might do, and the risk of each.

## Inference and the base model — what you get after pre-training

**Inference** means generating text from a trained model. It is a loop:

1. Feed in the tokens so far (the _prefix_), e.g. token 91.
2. Get the probabilities for the next token.
3. Pick one at random, weighted by those probabilities — like flipping a biased coin. Say we get 860.
4. Append it and go back to step 1.

Because step 3 is random, the same prompt gives a different answer each time. The output is a _remix_ of the training data: statistically similar, but usually not copied word for word. When you chat with ChatGPT, it is only doing inference. The parameters were fixed months earlier; nothing is learned during your chat.

**Two real examples:**

| Model | Year | Parameters | Context length | Training tokens |
| --- | --- | --- | --- | --- |
| GPT-2 (OpenAI) | 2019 | 1.6 billion | 1,024 | ~100 billion |
| Llama 3.1 (Meta) | 2024 | 405 billion | much longer | 15 trillion |

Training GPT-2 cost about $40,000 in 2019. Karpathy reproduced it in about 1 day for about $600, and thinks $100 is possible today. Costs fell because of better data, faster hardware and better software.

To release a model you need two things: the code for the network (a few hundred lines, fairly standard) and the parameters (one long list of numbers — the valuable part).

**What a base model is like.** The model at the end of pre-training is a _base model_: an internet-text simulator, a very expensive autocomplete. Using Llama 3.1 405B base (hosted by Hyperbolic), Karpathy shows:

- **It is not an assistant.** Ask "What is 2+2?" and it may answer, or wander off into philosophy, or write more questions. It is just continuing a web page.
- **It holds a lot of knowledge,** like a lossy zip file of the internet. Start a "top 10 landmarks in Paris" list and it fills it in. Things seen often are remembered better than rare things.
- **It can recite.** Paste the first sentence of Wikipedia's _Zebra_ page and it continues almost word for word for a long stretch. High-quality sources are shown to the model several times (maybe ~10), so it memorizes them. This is called _regurgitation_ and is usually unwanted.
- **It guesses past its knowledge cutoff.** Llama 3's data ends in 2023. Prompt it about the 2024 US election and it makes up different "parallel universes" each time (Pence as running mate against Clinton; DeSantis against Biden). This is _hallucination_.
- **It can learn from examples in the prompt** (_in-context learning_). Give 10 English→Korean word pairs, then "teacher:", and it gives the right translation. This is a _few-shot prompt_.
- **You can fake an assistant.** Write a prompt that looks like a web page with a conversation between a helpful AI and a human, then add your question. The model plays the assistant role — though it may then keep going and invent the human's next question too.

## Stage 2: Supervised fine-tuning — turning the base model into an assistant

To get an assistant, we keep training the base model with the exact same algorithm, but swap the internet data for a dataset of **conversations**. The model quickly learns how "the assistant" talks. Pre-training might take ~3 months on thousands of computers; this stage might take ~3 hours, because the dataset is much smaller.

You cannot program the assistant's behavior in code. You program it **by example**: helpful answers, a friendly tone, and refusals for things it should not help with all come from examples in the data.

**Conversations become tokens too.** Each model has a chat format. GPT-4o's looks roughly like this:

```text
<|im_start|>user<|im_sep|>What is 2+2?<|im_end|>
<|im_start|>assistant<|im_sep|>2+2 = 4<|im_end|>
```

`<|im_start|>`, `<|im_sep|>` and `<|im_end|>` are new special tokens added in this stage. The whole two-turn example is just a sequence of 49 tokens, so we can train on it like any other text. When you chat, the server builds this sequence from your messages, ends it with `<|im_start|>assistant<|im_sep|>`, and lets the model continue.

**Where the conversations come from.** In OpenAI's InstructGPT paper (2022), paid human contractors wrote both prompts ("List five ideas to regain enthusiasm for my career") and ideal answers. They followed labeling instructions from the company — in short: be **helpful, truthful and harmless** — which in practice run to hundreds of pages. Today most of this data is made with help from other LLMs and then edited by people; modern datasets like UltraChat have millions of mostly synthetic conversations.

**So what are you talking to?** Not a magical AI that researched your question. Roughly, you are getting a **statistical simulation of a skilled human labeler** following the company's instructions. Ask for the top Paris landmarks: if that question is in the training data, you will likely get something close to what that labeler wrote after ~20 minutes of research. If not, the model mixes its pre-training knowledge with the labeler style it learned.

## LLM psychology — hallucinations, memory and "self"

### Why models hallucinate

In the training data, questions like "Who is Tom Cruise?" or "Who is John Barrasso?" are always answered confidently and correctly. So the model learns the _style_: "Who is X?" → a confident biography. Ask about a made-up name like "Orson Kovacs" and an old model (Falcon 7B) happily says he is a sci-fi writer — then, on retry, a 1950s TV character, then a minor-league baseball player. It does not know, so it samples a plausible-sounding answer each time.

**Fix 1: teach it to say "I don't know".** Meta did this for Llama 3:

1. Take a paragraph from a training document (e.g. about hockey player Dominik Hašek).
2. Use an LLM to write factual questions and answers from it ("How many Stanley Cups did he win?" → 2).
3. Ask the model each question several times and have another LLM check the answers.
4. If it is often wrong (it said 4, then 0), add a training conversation where the right answer is "I'm sorry, I don't know."

The idea: somewhere inside, the network probably already has a signal for "I'm not sure". These examples teach it to connect that signal to actually saying so.

**Fix 2: let it look things up (tools).** Add special tokens like `<SEARCH_START>` and `<SEARCH_END>`. When the model writes a query between them, the program running it pauses, runs a web search, and pastes the results into the conversation. Then the model continues with the facts right in front of it. A few thousand training examples are enough to teach this. That is what you see when ChatGPT shows "Searching the web" and adds citations.

### Two kinds of memory

This is a key mental model:

|  | Where it lives | Like for a human |
| --- | --- | --- |
| Knowledge in the **parameters** | Weights, from training | A vague memory of something you read a month ago |
| Knowledge in the **context window** | The tokens of the current chat | Working memory: what is right in front of you |

Practical tip: don't ask "Summarize chapter 1 of _Pride and Prejudice_" and rely on memory. Paste the chapter into the prompt. The summary will be clearly better, just as yours would be if you re-read the chapter first.

### The model has no real "self"

Asking a model "Who are you, who built you?" is a bit nonsensical. It has no persistent existence: it starts up, processes tokens, and is wiped after each conversation. Without special training, an old model like Falcon might say "I was built by OpenAI" — not because it was trained on OpenAI data, but because ChatGPT is all over the internet, so that is the most likely answer.

Companies set the identity in two ways:

- **Hard-coded training examples.** The open-source OLMo 2 model's dataset includes 240 conversations like "Tell me about yourself" → "I am OLMo, an open language model developed by Ai2…".
- **A system message.** Hidden text at the start of every chat: "You are ChatGPT-4o, made by OpenAI, knowledge cutoff …". You see a blank page, but those tokens are in the context window.

Either way, the identity is bolted on, not deeply there like in a person.

## Models need tokens to think

Each new token is produced by one pass through the network. A modern network has maybe ~100 layers, so the amount of computation per token is **small and roughly fixed**. The model cannot do an arbitrarily hard calculation inside a single token. Real reasoning has to be **spread across many tokens**.

**Example.** "Emily buys 3 apples and 2 oranges. Each orange costs $2. The total is $13. What does each apple cost?" Two training answers both end in $3:

- **Bad label:** "The answer is $3. This is because 2 oranges cost $4, 13 − 4 = 9, 9 / 3 = 3." The model must produce "3" in a single token, with no work done yet. Everything after it is just justification after the fact.
- **Good label:** "Oranges cost 2 × $2 = $4. Apples cost $13 − $4 = $9. Each apple is $9 / 3 = $3." Each token only does a small step, and earlier results sit in the context for later steps.

Those step-by-step lines in ChatGPT's answers are not just for you — they are for the model. Karpathy's test: forced to answer in one token, the model got the easy version right, but with bigger numbers (23 apples, 177 oranges) it said 5 instead of 7. Allowed to work step by step, it got 7.

**Use tools for exact work.** Mental arithmetic in a network can slip at any step. Saying "use code" makes the model write a short Python program; a real interpreter runs it and the result goes back into the context. You can check the code, and Python does not make arithmetic mistakes.

**Counting and spelling are weak spots,** for two reasons:

1. **Counting** asks for too much in one token. "How many dots are below?" (177 dots) — the model guessed 161. With "use code", it just copies the dots into a Python string and calls `.count()` — copying is easy for it.
2. **Models do not see letters, only tokens.** "ubiquitous" is 3 tokens to the model. So "print every 3rd character" fails, and "How many r's in strawberry?" famously got "2" for a long time — it combines both weaknesses. Again, "use code" fixes it.

**The Swiss cheese model.** Models can solve Olympiad math and PhD-level science questions, yet sometimes fail at something trivial — there are holes in the cheese. Example: some models insist **9.11 is bigger than 9.9**. Researchers who looked inside found neurons linked to **Bible verses** lighting up; in Bible verse numbering, 9:11 comes after 9:9. It is not fully understood.

Takeaway: treat the model as a powerful but random-in-places tool. Use it, but check its work.

## Stage 3: Reinforcement learning — practice problems

**Why SFT is not enough.** For the apple problem, which written solution is best _for the model_? A human labeler cannot really know. A step that is easy for us might be too big a leap for the model; steps we spell out might be trivial for it and waste tokens. The model also knows things we don't, and vice versa. Our cognition is different from its cognition. So the model should **find for itself** which token sequences reliably reach the right answer.

**How RL works — guess and check:**

1. Take a prompt with a known final answer (e.g. $3).
2. Let the model generate many solutions — in practice thousands per prompt. Each is different, because sampling is random.
3. Check each one: did it reach $3? Say 4 of 15 did.
4. Train on the good ones (e.g. the best of the correct ones), so the model is a bit more likely to take those paths next time.
5. Repeat across tens of thousands of math, code and logic problems.

No human writes these solutions; they come from the model's own practice. SFT is still useful: it gets the model _near_ good solutions. RL then dials it in.

**What came out of it: thinking.** Pre-training and SFT have been standard for years. RL for LLMs is newer, and labs kept the details private. DeepSeek's **R1 paper** (early 2025) described it publicly. Two results stood out:

- Math accuracy kept rising over thousands of RL steps.
- Answers got **much longer**. Without anyone writing it in, the model started saying things like "Wait, wait. That's an aha moment. Let me re-evaluate this step by step." It learned to try different approaches, backtrack and double-check — a **chain of thought**, much like what happens in your head rather than what you write on paper.

These are called **reasoning** or **thinking models**. On the apple problem, R1 works it out, says "let me check my math again", re-solves it with an equation, and only then writes a clean answer for you. As of the video: DeepSeek R1 (open weights, also hosted by e.g. together.ai), OpenAI's o1 / o3-mini (which show only summaries of the thinking, partly to stop others copying it), and Google's Gemini 2.0 Flash Thinking. GPT-4o, by contrast, is mostly an SFT model. Karpathy uses GPT-4o for ~80–90% of everyday questions and switches to a thinking model for hard math and code.

**Why RL can beat humans: AlphaGo.** DeepMind's Go program showed this years earlier. A model trained only to imitate expert games improves but levels off below the best humans. A model trained with RL — playing itself and reinforcing moves that win — passed top player Lee Sedol. Its famous **move 37** had about a 1-in-10,000 chance of being played by a human, yet turned out brilliant.

With LLMs we are only seeing the first hints of this. In principle, RL could find new ways of thinking — new analogies, new strategies, maybe even its own language better than English for reasoning. That needs huge, varied sets of practice problems, and building those is a big part of current frontier research.

## RLHF — RL when there is no right answer

Math and code are _verifiable_: you can check the answer automatically. Jokes, poems and summaries are not. "Write a joke about pelicans" has no answer key.

You could have humans score every attempt, but the numbers explode. With 1,000 updates × 1,000 prompts × 1,000 attempts each, people would have to read **1 billion** pelican jokes.

**The trick (RLHF, Reinforcement Learning from Human Feedback):** train a separate neural network, a **reward model**, to imitate human taste, then do RL against it.

1. Generate, say, 5 jokes for a prompt.
2. A human **ranks** them best to worst. Ranking is easier than giving exact scores.
3. The reward model reads (prompt, joke) and outputs a score from 0 to 1. Train it so its scores agree with the human ranking. Example: if the human's #2 joke got only 0.1, nudge it up; if the worst joke got a high score, nudge it down.
4. Now run RL against the reward model, which can be asked as many times as you want.

Humans only look at maybe ~5,000 jokes instead of a billion.

**Upside.** It works in any domain, and it makes models measurably better. Karpathy's best guess why: it is much easier for people to **judge** than to **create**. Ranking 5 poems is easy; writing the ideal poem is hard. So the human signal is more accurate.

**Downside: the reward model can be gamed.** It is only a lossy simulation of a person — a big network with odd blind spots. Run RL long enough and it finds nonsense inputs that score high. In practice the pelican jokes improve for a few hundred steps, then fall off a cliff: the "best joke" becomes "the the the the the", which the reward model rates 1.0. These are _adversarial examples_. You can patch each one, but there are endlessly many.

So RLHF runs only for a limited number of steps, then you stop and ship. Karpathy's line: **RLHF is not "real" RL** — not the magical kind you can run forever. In Go, or a math problem with an answer key, the score cannot be faked, so RL can run indefinitely and find brand-new strategies. RLHF is more like a small fine-tune that makes the model a bit better. GPT-4o went through RLHF; that does not make it a thinking model.

## What's coming, and where to find models

**Coming next (as predicted in early 2025):**

- **Multimodal models.** Audio and images get turned into tokens too (slices of an audio spectrogram, small patches of an image), so the same methods apply. Models will hear, speak, see and draw natively.
- **Agents.** Models that work on long tasks for minutes or hours and report back. Humans become supervisors; people will talk about "human-to-agent ratios".
- **Computer use.** Models that click and type for you (OpenAI's Operator was an early example).
- **Learning after training.** Today the parameters are frozen after training; the only "learning" during use is what sits in the context window. Context windows are finite, so long multimodal tasks will likely need new ideas.

**Staying up to date:** LM Arena (a leaderboard from blind human comparisons — useful, though Karpathy thinks it has become somewhat gamed), the AI News newsletter, and following trusted people on X.

**Where to run models:**

| You want | Go to |
| --- | --- |
| Big proprietary models | The provider's site: ChatGPT, Gemini, etc. |
| Open-weights models (DeepSeek, Llama) | An inference provider such as together.ai |
| A base model | Hyperbolic (serves Llama 3.1 405B base) |
| A model on your own laptop | LM Studio, with smaller "distilled" models at lower precision |

## Putting it together — what happens when you hit Enter

1. Your message is put into the chat format and becomes a sequence of tokens.
2. The model continues the sequence, one token at a time, sampling from probabilities.
3. **Its knowledge** comes from pre-training on the internet — a vague, lossy memory.
4. **Its personality and answer style** come from SFT. For a normal model like GPT-4o, the answer is best seen as a neural-network simulation of a skilled company labeler following the labeling instructions — produced in seconds, with a small fixed amount of thinking per token.
5. **For a thinking model** (o3, R1), there is something more: reasoning strategies the model discovered for itself through RL. These shine in verifiable areas like math and code. Whether they carry over to writing and other open-ended tasks is still an open question.

Because of how they are made, models hallucinate, slip on mental math, can't easily count or spell, and have random holes like 9.11 vs 9.9. Karpathy's advice: use them every day as tools — for inspiration, first drafts and questions — but **check their work and own the result**.

## Answers to "Check yourself"

1. Dropped, because 0.60 is below 0.65. The score is the classifier's confidence, not a word count. Mixed languages, very short text, or pages full of names, numbers or code all lower that confidence, even when most words are English.
2. Vietnamese pages score close to 0 on the English filter, so nearly all are thrown away and the model rarely sees Vietnamese. Fixes: pre-train on multilingual data such as FineWeb-2 with a set share of Vietnamese; add Vietnamese SFT conversations; use a tokenizer that splits Vietnamese efficiently. SFT alone teaches the answer style but adds little knowledge.
3. List: 104 101 256 111 32 104 101 256 111, so 9 symbols. Vocabulary: 0–255 plus 256 = 257. The single "l" (108) stays in the vocabulary.
4. Bytes can already write any text, so nothing is missing. But the lists get long ("hello world" = 11 bytes vs 2 tokens), which fills the context faster and costs more compute. A bigger vocabulary gives shorter lists, but a bigger token table (100,277 × 768 ≈ 77M parameters) and rare tokens that are learned less well.
5. "t" = 1 byte (116), "ñ" = 2 (195 177), 😀 = 4 (240 159 152 128). Yes: every byte is a value from 0 to 255.
6. For example, inputs "cat sat on the" and targets "sat on the mat" (the inputs shifted by one). That gives 4 guesses: cat → sat, cat sat → on, cat sat on → the, cat sat on the → mat.
7. Only "cat" and the tokens before it in the window. Without the mask, it could read the next token (the answer) and copy it. It would learn nothing useful for writing text, where the future does not exist yet.
8. 16 × 4 = 64 guesses. The parameters are adjusted once, using the average error of all 64.
9. −ln: 0.69, 1.39, 2.30, 0.00. Sum 4.38, mean ≈ 1.10.
10. About 1/100,277 ≈ 0.001% each, close to equal odds.
11. Reject with a "context length exceeded" error: you get no answer. Keep only the last ~1,000 tokens: the answer can sound confident while missing most of the text. Split into chunks, or search for the relevant parts (RAG): extra steps, and links between distant parts can be missed.

**Source:** Andrej Karpathy, [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) (YouTube, 2025).
