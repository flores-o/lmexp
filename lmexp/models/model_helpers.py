MODEL_LLAMA_3_CHAT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_GPT2 = "openai-community/gpt2"
MODEL_QWEN_15 = "Qwen/Qwen1.5-7B-Chat"
MODEL_GEMMA_2 = "google/gemma-2-9b-it"

MODEL_ID_TO_END_OF_INSTRUCTION = {
    MODEL_LLAMA_3_CHAT: "<|start_header_id|>assistant<|end_header_id|>",
    MODEL_QWEN_15: "<|im_start|>assistant",
    MODEL_GEMMA_2: "<start_of_turn>model",
}
