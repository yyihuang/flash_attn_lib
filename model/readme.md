# Model Analysis for Flexllm Demo 

Model: Llama-2-70b-hf
https://huggingface.co/meta-llama/Llama-2-70b-hf

```
( vocab_size = 32000hidden_size = 4096intermediate_size = 11008num_hidden_layers = 32num_attention_heads = 32num_key_value_heads = Nonehidden_act = 'silu'max_position_embeddings = 2048initializer_range = 0.02rms_norm_eps = 1e-06use_cache = Truepad_token_id = Nonebos_token_id = 1eos_token_id = 2pretraining_tp = 1tie_word_embeddings = Falserope_theta = 10000.0rope_scaling = Noneattention_bias = Falseattention_dropout = 0.0mlp_bias = Falsehead_dim = None**kwargs )
```
* hdim=128

Model

To dive deeper into the model architecture and model loading, here is an example: https://zhuanlan.zhihu.com/p/9912733791 from Chenyang Zhao [Loading Llama-3.2-1B into SGLang]

