# AI-assisted Brainstorming Tool

## Overview

This is a command line prototype for an AI-assisted brainstorming tool that uses LLMs to generate and evaluate ideas for a given problem statement, surfacing the most promising ideas at the end.

## Key Features

- Tree-based structure to capture lineages of idea iterations
- Utilizes LLMs to generate ideas and evaluate them against multiple criteria
- Implements a Monte Carlo Tree Search algorithm to explore the idea space
- Configurable to use different LLM providers ([OpenAI](https://platform.openai.com/), [OpenRouter](https://openrouter.ai/), [Groq](https://www.groq.com/))
- Customizable problem statements, constraints for ideas, and idea evaluation criteria

## How It Works

1. The user defines a problem statement, constraints, and criteria for idea evaluation.
2. The system generates seed ideas using an LLM, employing various creative strategies.
3. Ideas are evaluated by an LLM and scored based on various criteria of your choosing.
4. A tree structure is built to represent idea lineages.
5. Monte Carlo Tree Search is used to explore and expand promising idea branches.
6. The process iterates to generate a diverse set of potential solutions, revealing the most promising ideas at the end.

## Tech Stack

- Python
- OpenAI Python SDK (you can use OpenAI, OpenRouter, or local models with Ollama or LM Studio or similar)
- Groq Python SDK

## Setup and Usage

1. Clone the repository:
   ```
   git clone https://github.com/mikecreighton/ai-brainstorm.git
   cd ai-brainstorm
   ```

2. Create a virtual environment, and activate it. Here's an example using `venv`.
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp .env.template .env
   ```
   Then open the `.env` file and add your API keys and specify the LLM provider. Possible values are `openai`, `openrouter`, and `groq`.

5. Choose your models. Because each provider has different models, you will need to specify the models you want to use. Change the `LLM_GENERATION_MODEL` and `LLM_EVALUATION_MODEL` variables at the top of `main.py` accordingly. The defaults models have been chosen based on their parameter size in an attempt to balance performance and cost.

6. Customize the problem statement, constraints, and evaluation criteria:
   Open `main.py` and scroll to the bottom. Look for the section where these are defined and modify them according to your needs.

7. Run the idea generation process:
   ```
   python main.py
   ```

8. (Optional) To continue a previous run, use:
   ```
   python main.py <previous_run.json> continue
   ```
   Replace `previous_run.json` with the actual filename of your previous run.

## Runtime Notes

As the Monte Carlo Tree Search algorithm explores the idea space, it will generate a large number of ideas. With each idea generation, the tree will be saved to a JSON file. You can continue a previous run by passing the filename of the previous run to the script along with the `continue` argument.

If you don't pass the `continue` argument, the script will simply show you the top ideas from your tree and then exit. Which you probably gathered from reading the code.

## Background

This idea is inspired by a couple research papers that explore techniques for improving reasoning and creativity in LLMs:

- RAP: [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992)
- rStar: [Mutual Reasoning Makes Smaller LLMs Stronger Problem-solvers](https://huggingface.co/papers/2408.06195)
- Tree of Thoughts (ToT): [Tree of thoughts: Deliberate problem solving with large language models.](https://arxiv.org/abs/2305.10601)

I didn't implement all of that. Instead, I cherry-picked some ideas and combined them into a single tool. Since all of these papers were inherently about wide iteration and convergence on a correct solution, I thought the same idea could be applied to brainstorming, where you want to generate a wide range of ideas and then surface the best ones.

So here are the bits that I felt were relevant to the activity of brainstorming:

1. Using small language models (instead of SOTA frontier models) to maximize the ratio of number of ideas to cost
2. Monte Carlo Tree Search ([MCTS](https://link.springer.com/chapter/10.1007/11871842_29)) as a way to explore the idea space, balancing exploration and exploitation
3. Iterativly improving on ideas based on a reward design similar to [self-evaluation methods](https://arxiv.org/abs/2211.00053) that use natural language to evaluate ideas
4. Combining MCTS with small language models in order to improve the quality of output for those small language models
5. Using a defined set of "actions" (in this implementation, "creative directives") to generate each iteration step of an idea

Ultimately, SOTA models like Claude 3.5 Sonnet and OpenAI's GPT-4o generate better zero-shot ideas than small language models (assuming you're good at prompting and really know how to push a model). But they are also _way_ more expensive, especially when you're pushing for quantity. So we might be sacrificing some quality for cost. But it's more valuable to have a lot of ideas than to have a few good ones. Especially since we're just looking for an idea to spark something in us.

Also, I haven't done a 1-to-1 comparison within this architecture of the quality of ideas generated by the small language models vs. the SOTA models because it would be so expensive to run. With this tool, it's possible to use ~8B parameter models and generate 100s of ideas for less than 10 cents.

## Other Notes

- There are a lot of print statements in the code because it's running in a terminal, and it's fun to see what's being generated in real time. But that can make the code hard to read. So delete them if you want to.
- The LLM responses from idea evaluation are quite subjective, and only happen once per idea. So the quality of the ideas generated is only as good as the LLM's ability to evaluate them. Multiple evaluations on a single idea from different models might be a good thing to try.
= There are so many other ways to iterate on this tool — prompt design (introducing few-shot prompting), action design, discriminator addition and design, and evaluation methods. Any number of these could improve creativity of the ideas that are generated.

## Credits

- Claude for helping me formalize my brainstorming idea by suggesting ways to integrate some of the research methods I've been reading about.
- Claude (Claude 3.5 Sonnet), ChatGPT (GPT-4o), and Cursor (_especially_ Cursor for the AI-assisted code writing), helped me make this. I'm not a computer scientist, so implementing the Monte Carlo Tree Search algorithm was a bit of a challenge conceptually. But now I've got a handle on it.
- All the cool research work (linked above) that's been done on improving reasoning in LLMs.

## License

MIT License. See `LICENSE` for details.
