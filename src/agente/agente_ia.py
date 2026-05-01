import logging
import torch

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from functools import lru_cache

from langchain_huggingface import HuggingFacePipeline
# from langchain_community.llms import HuggingFacePipeline
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

from langchain.agents import create_agent
#from langchain.agents import AgentExecutor

from src.agente.agent_tools import _stock_lookup, _technical_analysis, _price_prediction, _rag_knowledge



logger = logging.getLogger(__name__)

REACT_PROMPT = """Você é um analista financeiro especializado em previsão de preços de ações.

        Use as ferramentas disponíveis para responder perguntas sobre ações,
        previsões e análises técnicas.

        Se precisar de dados, use as ferramentas.
        Responda de forma clara e objetiva."""

@lru_cache(maxsize=1)
def load_model(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quantize: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Carrega modelo Qwen com quantização INT4 via bitsandbytes.

    Args:
        model: modelo HuggingFace Hub.
        quantize: aplica quantização NF4.

    Returns:
        Tupla (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # if quantize:
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_use_double_quant=True,
    #     )
    #     model_llm = AutoModelForCausalLM.from_pretrained(
    #         model,
    #         quantization_config=bnb_config,
    # #        device_map="auto",
    #         trust_remote_code=True
    #     )
    # else:
    #     model_llm = AutoModelForCausalLM.from_pretrained(
    #         model,
    #         torch_dtype=torch.float16,
    # #        device_map="auto",
    #         trust_remote_code=True,
    #     )

    model_llm = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    model_llm.eval()
    return model_llm, tokenizer


def generate_response(
    prompt: str,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens: int = 206,
    temperature: float = 0.3,
    quantize: bool = True,
) -> str:
    """
    Args:
        prompt: Texto da oergunta.
        model: modelo no huggingFace.
        max_new_tokens: Máximo de tokens gerados.
        temperature: Temperatura de sampling.
        quantize: modelo quantizado.
    """
    model_llm, tokenizer = load_model(model, quantize)

    messages = [
        {"role": "system", "content": "Você é um analista financeiro especializado."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model_llm.device)

    with torch.inference_mode():
        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)

def build_llm(model: str = "Qwen/Qwen2.5-0.5B-Instruct", quantize: bool = True) -> HuggingFacePipeline:
    """Constrói agente llm para uso com LangChain."""
    model_llm, tokenizer = load_model(model, quantize)
    pipe = pipeline(
        "text-generation",
        model=model_llm,
        tokenizer=tokenizer,
        max_new_tokens=206,
        temperature=0.3,
        do_sample=True,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


def build_agent(
    tools: list[Tool],
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quantize: bool = False,
):
    """Cria agente ReAct para o Datathon.

    Args:
        tools: Lista de ferramentas.
        model: Modelo LLM a utilizar.
        quantize: Se True, usa quantização INT4.
    """

    llm = build_llm(model, quantize)
    agent = create_agent(llm, tools=tools, system_prompt=REACT_PROMPT)

 #   agent_llm = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=20, handle_parsing_errors=True)

    return agent