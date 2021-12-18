import random
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from transformers import BartForConditionalGeneration, BartConfig
from transformers import PreTrainedTokenizerFast

from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

# Kobart의 attention layer를 대체
class LongformerSelfAttentionForBart(nn.Module):
    def __init__(self, config : dict , layer_id : int):
        super().__init__()
        self.embed_dim = config.d_model
        self.longformer_self_attn = LongformerSelfAttention(config, layer_id=layer_id)
        self.output = nn.Linear(self.embed_dim, self.embed_dim)

    # kobart의 기존 layer와 동일한 형태의 입력을 받고, 동일한 형태의 출력을 할 수 있도록 해줘야함.
    def forward(self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # bs x seq_len x seq_len -> bs x seq_len 으로 변경
        attention_mask = attention_mask.squeeze(dim=1)
        attention_mask = attention_mask[:,0]

        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        outputs = self.longformer_self_attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=None,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )

        attn_output = self.output(outputs[0])

        return (attn_output,) + outputs[1:] if len(outputs) == 2 else (attn_output, None, None)

class LongformerBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            self.model.encoder.embed_positions = BartLearnedPositionalEmbedding(
                config.max_encoder_position_embeddings, 
                config.d_model, 
                config.pad_token_id)

            self.model.decoder.embed_positions = BartLearnedPositionalEmbedding(
                config.max_decoder_position_embeddings, 
                config.d_model, 
                config.pad_token_id)
            
            for i, layer in enumerate(self.model.encoder.layers):
                layer.self_attn = LongformerSelfAttentionForBart(config, layer_id=i)

#longformer bart모델의 config 생성 class
class LongformerBartConfig(BartConfig):
    def __init__(self, attention_window: List[int] = [512], attention_dilation: List[int] = [1],
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks',
                 gradient_checkpointing: bool = False, max_seq_len: int = 4096, max_pos: int = 4104,  **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """

        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        self.gradient_checkpointing = gradient_checkpointing
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2']


if __name__ == '__main__':
	max_pos = 4104
	max_seq_len = 4096
	attention_window = 512
	save_path = 'longformer_kobart'

	# 기존 pretrained 된 kobart tokenizer & model load
	tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1', model_max_length=max_pos)
	kobart_longformer = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
	config  = LongformerBartConfig.from_pretrained('gogamza/kobart-base-v1')

	kobart_longformer.config = config

	config.attention_probs_dropout_prob = config.attention_dropout
	config.architectures = ['LongformerEncoderDecoderForConditionalGeneration', ]


	# Tokenizer의 max_positional_embedding_size 확장
	# extend position embeddings
	tokenizer.model_max_length = max_pos
	tokenizer.init_kwargs['model_max_length'] = max_pos
	current_max_pos, embed_size = kobart_longformer.model.encoder.embed_positions.weight.shape
	assert current_max_pos == config.max_position_embeddings + 2

	config.max_encoder_position_embeddings = max_pos
	config.max_decoder_position_embeddings = config.max_position_embeddings
	del config.max_position_embeddings
	max_pos += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
	assert max_pos >= current_max_pos

	new_encoder_pos_embed = kobart_longformer.model.encoder.embed_positions.weight.new_empty(max_pos, embed_size)

	# Positional Embedding 확장
	k = 2
	step = 1028 - 2
	while k < max_pos - 1:
		new_encoder_pos_embed[k:(k + step)] = kobart_longformer.model.encoder.embed_positions.weight[2:]
		k += step
	kobart_longformer.model.encoder.embed_positions.weight.data = new_encoder_pos_embed

	config.attention_window = [attention_window] * config.num_hidden_layers
	config.attention_dilation = [1] * config.num_hidden_layers

	# Kobart Self attention > Longformer Self Attention
	for i, layer in enumerate(kobart_longformer.model.encoder.layers):
		longformer_self_attn_for_bart = LongformerSelfAttentionForBart(kobart_longformer.config, layer_id=i)

		longformer_self_attn_for_bart.longformer_self_attn.query = layer.self_attn.q_proj
		longformer_self_attn_for_bart.longformer_self_attn.key = layer.self_attn.k_proj
		longformer_self_attn_for_bart.longformer_self_attn.value = layer.self_attn.v_proj

		longformer_self_attn_for_bart.longformer_self_attn.query_global = copy.deepcopy(layer.self_attn.q_proj)
		longformer_self_attn_for_bart.longformer_self_attn.key_global = copy.deepcopy(layer.self_attn.k_proj)
		longformer_self_attn_for_bart.longformer_self_attn.value_global = copy.deepcopy(layer.self_attn.v_proj)

		longformer_self_attn_for_bart.output = layer.self_attn.out_proj

		layer.self_attn = longformer_self_attn_for_bart

	kobart_longformer.save_pretrained(save_path)
	tokenizer.save_pretrained(save_path, None)