---
title: "How to Use LLMs Like ChatGPT: An Easy Guide"
date: "2026-11-19"
topic: "AI"
tags: ["LLM", "ChatGPT", "Tools", "Prompting", "Karpathy"]
summary: "Andrej Karpathy's \"How I use LLMs\" in plain language: what you are actually talking to, when to use thinking models, search, code and file uploads, and why everything an LLM gives you is a first draft."
---

## The big idea

_Watch:_ [0:00 · Intro into the growing LLM ecosystem](https://youtu.be/EWvNQjAaOHw?t=0)

A chatbot like ChatGPT is, at its core, a big compressed file of the internet that predicts the next piece of text. Everything else — search, code, voice, images — is a tool bolted on around it.

This article is a plain-language write-up of Andrej Karpathy's video [How I use LLMs](https://www.youtube.com/watch?v=EWvNQjAaOHw) (early 2025). His earlier video explained how these models are built; this one is practical: which features exist, when to use each, and what to watch out for.

Note: the video was filmed in early 2025. Model names, prices and which app has which feature change fast, so treat those details as examples, not current facts. The way of thinking is what lasts.

Each section has a **Watch** link that opens the video at the start of that chapter, using the chapter times from the video's description. The point discussed is somewhere inside that chapter.

## What you are actually talking to

_Watch:_ [2:54 · ChatGPT interaction under the hood](https://youtu.be/EWvNQjAaOHw?t=174)

The chat bubbles are an illusion. Under the hood, you and the model are writing into one long line of text chunks called **tokens**.

### Tokens

Text is cut into small pieces (tokens). A piece can be a word, part of a word, or a space plus a word. GPT-4o has a vocabulary of about 200,000 possible tokens.

Concrete example from the video: "Write a haiku about what it's like to be a large language model" = **15 tokens**. The model's haiku reply = **19 tokens**. With the hidden special tokens that mark "user starts / user ends / assistant starts", the whole exchange = **42 tokens**.

You can see this yourself with a site like Tiktokenizer.

### The context window = working memory

1. You click **New chat** → the token line is empty.
2. You type → your tokens go into the line.
3. You press Enter → the model adds its own tokens until it emits a special "I'm done" token.
4. Control comes back to you, and you take turns.

That growing line is the **context window**. Anything inside it is in the model's working memory and easy for it to use. New chat wipes it.

### How the model was made

1. **Pre-training** — the model reads a huge part of the internet and learns to predict the next token. The result is like a **lossy, probabilistic zip file** of the internet: roughly 1 TB ≈ 1 trillion parameters. It keeps the gist, not exact copies. This costs tens of millions of dollars and ~3 months, so it is done rarely.
2. **Post-training** — the model is trained on example conversations written by humans. This gives it the _assistant personality_: it answers questions instead of just continuing internet pages.

Two results follow:

- **Knowledge cutoff.** The model only knows what was on the internet when pre-training ended. Last week's news is not in it.
- **Vague memory.** Things written about a lot online (e.g. caffeine in an espresso shot, ~63 mg) are remembered well. Rare things are remembered poorly and more likely made up.

### Karpathy's one-paragraph summary

> Hi, I'm ChatGPT. I'm a one-terabyte zip file. My knowledge comes from the internet, which I read in full about six months ago and only remember vaguely. My winning personality was programmed by example by human labelers at OpenAI.

By default it has **no tools**: no calculator, no web, no code runner. Just tokens in, tokens out.

### When is it fine to trust it without tools?

_Watch:_ [13:12 · Basic LLM interactions examples](https://youtu.be/EWvNQjAaOHw?t=792)

His rule: the info is (a) **not recent**, (b) **common on the internet**, and (c) **not high stakes**. Example: asking which ingredients in DayQuil/NyQuil help a runny nose. He still checked the box to confirm. Always treat answers as a good first draft, not guaranteed truth.

## Three good habits

### 1. Start a new chat when you change topic

_Watch:_ [13:12 · Basic LLM interactions examples](https://youtu.be/EWvNQjAaOHw?t=792) (the new-chat tip comes near the end of this chapter)

Tokens in the context window cost you in two ways:

- **Distraction.** Old, unrelated text can confuse the model and lower answer quality.
- **Speed and cost.** More tokens = slightly slower and more expensive to produce each new token.

Keep the window short and on-topic. Keep old text only if it helps the current task.

### 2. Know which model you are using

_Watch:_ [18:03 · Be aware of the model you're using, pricing tiers](https://youtu.be/EWvNQjAaOHw?t=1083)

Bigger models know more, write better and make up fewer facts. Smaller models are cheaper and dumber. Companies charge more for the big ones. ChatGPT tiers at the time of the video:

| Tier | Price / month | What you got |
| --- | --- | --- |
| Free | $0 | Mostly GPT-4o mini (a smaller model) |
| Plus | $20 | GPT-4o, ~80 messages per 3 hours |
| Pro | $200 | Unlimited GPT-4o, o1 pro, Deep Research |

If you are logged out, you may not even be told which model answers. If you use these tools for serious work, paying for the top model is often worth it. Other apps (Claude, Gemini, Grok) have their own tiers and model pickers.

### 3. Use an "LLM council"

_Watch:_ [18:03 · Be aware of the model you're using, pricing tiers](https://youtu.be/EWvNQjAaOHw?t=1083) (the council idea comes near the end of this chapter)

Karpathy pays for several apps and asks them all the same question — e.g. "a cool city to visit". (Both Claude and Gemini suggested Zermatt, and he went.) Comparing answers shows where they agree and where one is off.

The main players he mentions: **ChatGPT** (OpenAI, the first and most feature-rich), **Gemini** (Google), **Copilot** (Microsoft), **Meta AI**, **Claude** (Anthropic), **Grok** (xAI), **DeepSeek** (China), **Le Chat** (Mistral, France). Leaderboards like Chatbot Arena and Scale's SEAL rank them.

## Thinking models

_Watch:_ [22:54 · Thinking models and when to use them](https://youtu.be/EWvNQjAaOHw?t=1374)

Thinking models spend extra time "thinking out loud" before answering. This makes them much more accurate on hard math, code and logic — but slower.

### Where they come from

There is a third training stage after pre- and post-training: **reinforcement learning (RL)**. The model practises on many math and code problems. It is rewarded when it gets the right answer, and through that it discovers its own thinking strategies: trying ideas, backtracking, rechecking assumptions. These strategies look a lot like a human's inner monologue. Humans could not easily write them down as examples, so RL is how the model finds them. DeepSeek's R1 paper was the first to describe this openly.

### Real example: a bug in his code

Karpathy had a gradient-check bug in a neural network script.

- **GPT-4o (no thinking):** gave a list of generic things to check. Did not find the bug.
- **o1 pro (thinking, $200 tier):** thought for ~1 minute and found it — the parameters were packed and unpacked in a mismatched order.
- Claude 3.5 Sonnet, Gemini and Grok 3 also found it, even without thinking. DeepSeek R1 (via Perplexity) found it after a few minutes of visible thinking.

### When to use one

- **Use thinking** for hard problems: math, code, tricky logic.
- **Skip it** for simple stuff like travel ideas. There is no point waiting a minute.
- His habit: **try the fast non-thinking model first**. If the answer looks weak, switch to a thinking model.

How to find it: OpenAI's models starting with "o" (o1, o3-mini…) were thinking models. Grok has a "Think" button. Claude 3.7 added an "extended" thinking mode.

## Tools, part 1: internet search and deep research

### How a tool works (the mechanism)

_Watch:_ [31:00 · Tool use: internet search](https://youtu.be/EWvNQjAaOHw?t=1860)

The model is trained to output a special token that means "search the internet for X". When the app sees it, it:

1. Pauses the model.
2. Runs the search and opens ~10 web pages.
3. Pastes their text into the context window (often thousands of tokens).
4. Lets the model continue — now it can read the answer from the pages.

So search doesn't make the model smarter. It puts fresh documents into its working memory.

### When to use search

Use it when the answer is **recent**, **changes over time**, or is **niche**. Rule of thumb: if you'd normally Google it and click the top few links, let the model do that. Karpathy's real queries (he uses Perplexity for these):

- Is the stock market open today? (It was Presidents' Day — closed.)
- When does White Lotus season 3 episode 2 come out? Where was it filmed?
- Does Vercel offer a Postgres database? (Offerings change.)
- What is Apple launching tomorrow?
- Why is Palantir stock going up?
- Is it safe to travel to Vietnam?
- What's the deal with this thing trending on Twitter?

Some models search on their own when they sense the question is recent. Others need you to click a "Search" button. Some (at the time: Claude, Gemini 2.0 Pro) had no search and just said "as of my last update…". Check results via the citations — search answers can still be wrong.

### Deep research = search + thinking, for a long time

_Watch:_ [42:04 · Tool use: deep research](https://youtu.be/EWvNQjAaOHw?t=2524)

Deep research runs for **~5–30 minutes**. It does many searches, reads many papers and pages, thinks, and writes a report with citations — like a custom research paper.

Example: he asked about **Ca-AKG** (2.5 g per serving in Bryan Johnson's longevity mix): why it's included, evidence in humans and animals, how it might work, safety concerns. ChatGPT thought for 5 minutes, read 27 sources, and reported on studies in worms, fruit flies and mice, ongoing human trials, mechanism and safety.

Other uses: comparing browsers for privacy (he switched to Brave), mouse life-extension research.

A failure example: a table of US LLM labs with size and funding. It left out xAI, wrongly included Hugging Face and EleutherAI, and some numbers looked off.

At the time, ChatGPT's version (Pro tier only) was the most thorough. Perplexity and Grok ("Deep Search") had similar but shorter versions.

**Rule: treat the report as a first draft.** It can still invent or misread things. Use the citations to go read the real sources.

## Giving it your own documents

_Watch:_ [50:57 · File uploads, adding documents to context](https://youtu.be/EWvNQjAaOHw?t=3057)

Uploading a file does the same thing as search: it puts real text into the context window, so the model reads it instead of relying on fuzzy memory.

### Reading papers

Drag a PDF into the chat and ask for a summary first, then ask questions as you read. His example: the Arc Institute paper on **Evo 2**, a model trained on DNA. He's not a biologist, so the LLM helped him follow it.

What probably happens under the hood: the PDF is turned into plain text and loaded into the context window. Images in it may be dropped or poorly understood.

### Reading books

Karpathy says he now rarely reads a book alone. Example: Adam Smith's _The Wealth of Nations_ (1776):

1. Get the chapter's text (old books are free on Project Gutenberg).
2. Paste it in: "We are reading The Wealth of Nations. Please summarize this chapter to start."
3. Read the chapter yourself and ask questions whenever you're confused.

Why paste it? The model knows the book exists, but doesn't remember each chapter exactly. Pasting it gives exact text.

He finds this greatly improves his understanding and memory, and makes hard texts (old language, other fields, Shakespeare) much less scary. His wish: a reader app where you highlight a passage and ask about it — he didn't know of a good one.

**Tip: don't read books alone.**

## Tools, part 2: running code

### The Python interpreter

_Watch:_ [59:00 · Tool use: Python interpreter, messiness of the ecosystem](https://youtu.be/EWvNQjAaOHw?t=3540)

Same trick as search: the model outputs special tokens meaning "run this program". The app runs the code and pastes the result back into the context window.

Why it matters: the model does math "in its head" by memory-like pattern matching, not real calculation.

- **30 × 9 = 270** — easy, the model just knows it. No tool needed.
- **A big multiplication** (two long numbers) — ChatGPT wrote Python and got it exactly right.
- Grok 3 had no code tool, did it in its head, and was very close — but a few digits were wrong.
- Claude wrote JavaScript instead, also correct. Gemini got one right in its head, then got a harder one wrong.

Lesson: **know which apps have a code tool.** Without one, the model will still give you a confident answer, which may be made up.

### Advanced Data Analysis (ChatGPT)

_Watch:_ [1:04:35 · ChatGPT Advanced Data Analysis, figures, plots](https://youtu.be/EWvNQjAaOHw?t=3875)

This turns ChatGPT into a **junior data analyst**: it collects data, writes code, and draws charts. His example:

1. "Search the web for OpenAI's valuation by year and make a table."
2. "Plot it with a log-scale y-axis." → chart drawn.
3. "Fit a trend line and extrapolate to 2030."

Two problems he caught by reading the code:

- **Silent assumption.** 2015 was "N/A", but the code quietly put in $0.1B (100 million).
- **Wrong summary.** It said the 2030 value was **~$1.7 trillion**. But the variable in the code showed **~$20 trillion**. When asked to print the variable, it admitted the mistake.

So: powerful, but **read and check the code**. Don't use it if you can't verify what it did.

### Claude Artifacts

_Watch:_ [1:09:00 · Claude Artifacts, apps, diagrams](https://youtu.be/EWvNQjAaOHw?t=4140)

Claude can write a small web app and run it right in the chat. Example: make 20 flashcards from the Adam Smith Wikipedia intro, then "use Artifacts to build a flashcard app". Claude wrote a React app with flip, correct/wrong and shuffle. No server or database — it runs in your browser.

His favourite use: **diagrams**. "Create a conceptual diagram of this chapter" → Claude writes Mermaid code and draws a tree of the chapter's argument (e.g. division of labour ↔ size of the market ↔ water vs land transport). Great for visual thinkers.

### Coding apps: Cursor and "vibe coding"

_Watch:_ [1:14:02 · Cursor: Composer, writing code](https://youtu.be/EWvNQjAaOHw?t=4442)

For real programming he does **not** paste code into a chat website — it lacks context. He uses an app like **Cursor** (others: VS Code, Windsurf) that works with the files on his computer and calls Claude through an API.

The feature levels grew over time:

- **Cmd+K** — change this line.
- **Cmd+L** — explain this code.
- **Cmd+I (Composer / agent)** — an autonomous agent that edits many files and runs commands.

Demo: with ~5 sentences he got a working Tic-Tac-Toe app. Then "add confetti when someone wins" → it installed a library, added styling, even added a victory sound.

**Vibe coding** (a term Karpathy coined) = give control to the agent, say what you want, and hope it works. If it breaks, you can still fall back to normal programming.

## Voice, images and video

The key idea: audio and images can also be cut into tokens. The core model doesn't care if a token is text, sound or picture — it just learns patterns in token streams. Only the encoder (in) and decoder (out) know the difference.

### Voice: "fake audio" vs "true audio"

_Watch:_ [1:22:28 · Audio (speech) input/output](https://youtu.be/EWvNQjAaOHw?t=4948) · [1:27:37 · Advanced Voice Mode, true audio](https://youtu.be/EWvNQjAaOHw?t=5257)

|  | Fake audio | True audio |
| --- | --- | --- |
| How it works | Speech → text → model → text → speech, using separate models | The model itself hears and speaks in **audio tokens** |
| Examples | Mic button in the ChatGPT app; Mac apps like SuperWhisper, WisprFlow, MacWhisper; "Read aloud" | ChatGPT **Advanced Voice Mode**; Grok voice mode |
| What it can do | Saves typing | Accents, voices (Yoda, pirate), speed, tone |

How audio becomes tokens: turn the sound into a spectrogram (frequencies over time), cut it into small windows, and map each to the closest of ~100,000 "audio chunks".

Karpathy speaks about **50%** of his queries on desktop and **~80%** on his phone. He binds F5 to SuperWhisper: press, talk, press, text appears. He types only for exact names (products, libraries) that transcribe badly.

He found ChatGPT's advanced voice too cautious (it refused to make a fox sound). Grok's voice has looser modes (romantic, unhinged, conspiracy…) if you want entertainment.

**NotebookLM** (Google): upload papers or pages, then press generate to get a custom ~30-minute **two-host podcast** about them. You can steer it with instructions and even interrupt with questions. Good for niche topics while walking or driving.

_Watch:_ [1:37:09 · NotebookLM, podcast generation](https://youtu.be/EWvNQjAaOHw?t=5829)

### Images in

_Watch:_ [1:40:20 · Image input, OCR](https://youtu.be/EWvNQjAaOHw?t=6020)

You can paste a picture and ask about it. Images are cut into a grid of patches, and each patch becomes a token.

His two-step method: **first ask it to transcribe** what's in the image into text (so you can check it read it right), **then ask your questions**. Examples:

- Nutrition label of a supplement → table of ingredients → rank them by how well studied/safe they are.
- Blood test results, screenshot by screenshot → interpretation. Fine as a first draft, but still talk to a doctor.
- A math formula from a paper → converted to text.
- Colgate toothpaste ingredients → which are needed, which are filler.
- A meme (crows: "attempted murder") → explained.

Mac shortcut: **Ctrl+Shift+Cmd+4** to copy a screen region, then paste into the chat.

### Images out

_Watch:_ [1:47:02 · Image output, DALL-E, Ideogram](https://youtu.be/EWvNQjAaOHw?t=6422)

ChatGPT used **DALL·E 3** (others: Ideogram). He uses it for YouTube thumbnails and icons. At the time, the chat model wrote a caption and a _separate_ image model drew it — not done natively.

### Video

_Watch:_ [1:49:14 · Video input, point and talk](https://youtu.be/EWvNQjAaOHw?t=6554) · [1:52:23 · Video output, Sora, Veo 2](https://youtu.be/EWvNQjAaOHw?t=6743)


In the ChatGPT phone app's advanced voice, you can turn on the camera and talk about what you see. It identified books, a CO₂ monitor (713 ppm — fine, aim below 800), and a Middle-earth map. It likely samples about one image per second. He suggests this for parents and grandparents: just point and ask.

Video _generation_ tools (e.g. Veo 2, Sora) are improving fast; he doesn't use them much.

## Quality-of-life features

All three of these work the same way underneath: **saved text that gets pasted into the start of the context window** for you.

### Memory

_Watch:_ [1:53:29 · ChatGPT memory, custom instructions](https://youtu.be/EWvNQjAaOHw?t=6809)

Normally each new chat starts empty. ChatGPT's memory saves short notes about you (e.g. "thinks late 1990s–early 2000s was peak Hollywood") into a list. That list is added to the start of every chat. Sometimes it saves on its own; sometimes you must say "please remember this". You can view, edit and delete the notes. Over time it gets to know you — he says it helps with things like movie picks.

### Custom instructions

_Watch:_ [1:53:29 · ChatGPT memory, custom instructions](https://youtu.be/EWvNQjAaOHw?t=6809) (custom instructions come after memory in this chapter)

A settings box for how you want it to behave in every chat. His:

- "Don't talk like an HR business partner — just talk normally."
- "Be educational; give explanations and insights."
- For Korean: use one specific level of politeness by default.

### Custom GPTs (saved prompts)

_Watch:_ [1:58:38 · Custom GPTs](https://youtu.be/EWvNQjAaOHw?t=7118)

A custom GPT is just a saved prompt. Nothing magical. It saves you from retyping a long setup each time. He mostly uses them for learning Korean:

1. **Vocabulary extractor** — give it a sentence; it returns words in dictionary form as `Korean; English`, ready to import into Anki flashcards.
2. **Detailed translator** — translates the full sentence, then breaks it down piece by piece (including small particles that Google Translate or Papago skip). You can ask follow-up questions. He thinks it beats dedicated translators.
3. **Korean captions** — paste a screenshot from a show (Singles Inferno) with burned-in subtitles; it does OCR, translates and breaks it down.

### Prompting tip: show examples (few-shot)

In each custom GPT he writes a short description **plus several worked examples** of input and desired output. He marks them with XML-like tags so the model knows where each example starts and ends.

- **Zero-shot** = describe the task only.
- **Few-shot** = describe the task and show examples.

Few-shot is almost always more accurate — just like teaching a person by showing, not only telling.

## Cheat sheet: what to reach for

_Watch:_ [2:06:30 · Summary](https://youtu.be/EWvNQjAaOHw?t=7590)

| You want to… | Use | Watch out for |
| --- | --- | --- |
| Ask about common, older knowledge | Plain chat, no tools | Vague memory; verify anything important |
| Solve hard math, code or logic | A thinking model | Slower; try the fast model first |
| Get recent or changing facts | Internet search | Check the citations |
| Get a full report on a topic | Deep research | Treat it as a first draft |
| Understand a paper or book | Upload or paste the text | Images in PDFs may be lost |
| Do exact math or make charts | Python / data analysis | Read the code; hidden assumptions |
| Build a small app or diagram | Claude Artifacts | Runs in browser only, no backend |
| Write real software | Cursor or similar | Review what the agent changed |
| Save typing | Voice input | Type exact names yourself |
| Ask about a photo or screenshot | Image upload | Ask it to transcribe first and check |
| Stop repeating yourself | Memory, custom instructions, custom GPTs | All just text added to the context |

Three things to always keep in mind:

1. **You're talking to a zip file.** Its knowledge is lossy and out of date. Tools fix that by putting real text into its working memory.
2. **Know your model and tools.** Different apps, tiers, and web vs phone versions all have different abilities. A model without a tool will still answer — and may make it up.
3. **Everything is a first draft.** Use it to go faster, then verify what matters.
