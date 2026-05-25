"""
LangChain ReAct Agent — 把 FinLoRA 微调模型 + 行情/指标工具串成一个完整的金融助手。

支持两种 LLM 驱动:
  A) 远程 API (OpenAI / DeepSeek / Qwen-DashScope) — 走 langchain-openai
  B) 本地 transformers HF pipeline — 走 langchain-huggingface (无需联网)

无论哪种，'analyze_sentiment' 工具内部用的都是我们自己微调的 FinLoRA Qwen 模型。
"""

from __future__ import annotations

import os

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from .prompts import AGENT_SYSTEM_PROMPT
from .tools import ALL_TOOLS


REACT_TEMPLATE = """{system}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: what to do next
Action: the action to take, exactly one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def _build_llm(backend: str):
    """根据 backend 选择 LLM。"""
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    if backend == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.environ.get("AGENT_MODEL", "deepseek-chat"),
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            temperature=0,
        )
    if backend == "local":
        # 直接复用 sentiment_llm 同款 Qwen 做 Agent 推理
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
        import torch

        model_name = os.environ.get("AGENT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        pipe = pipeline(
            "text-generation", model=m, tokenizer=tok,
            max_new_tokens=512, do_sample=False, return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=pipe)

    raise ValueError(f"unknown backend: {backend}")


def build_agent(backend: str = "openai", verbose: bool = True) -> AgentExecutor:
    """构建可执行的 ReAct Agent。"""
    llm = _build_llm(backend)
    prompt = PromptTemplate.from_template(REACT_TEMPLATE).partial(system=AGENT_SYSTEM_PROMPT)

    agent = create_react_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=verbose,
        max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def run_agent(question: str, backend: str = "openai") -> dict:
    executor = build_agent(backend=backend)
    return executor.invoke({"input": question})


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="用户问题")
    parser.add_argument("--backend", default=os.environ.get("AGENT_BACKEND", "openai"),
                        choices=["openai", "deepseek", "local"])
    args = parser.parse_args()

    out = run_agent(args.question, backend=args.backend)
    print("\n========== FINAL ANSWER ==========")
    print(out["output"])
    print("\n========== INTERMEDIATE STEPS ==========")
    for i, (action, obs) in enumerate(out.get("intermediate_steps", [])):
        print(f"\n[{i + 1}] Tool: {action.tool}")
        print(f"    Input: {action.tool_input}")
        print(f"    Obs:   {str(obs)[:200]}")
