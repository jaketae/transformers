# coding=utf-8
# Copyright 2022 The HuggingFace Team The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch FastSpeech2 model."""


import math
from typing import Optional

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ...activations import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_fastspeech2 import FastSpeech2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "microsoft/fastspeech2"
_CONFIG_FOR_DOC = "FastSpeech2Config"
_TOKENIZER_FOR_DOC = "FastSpeech2Tokenizer"

FASTSPEECH2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/fastspeech2",
    # See all FastSpeech2 models at https://huggingface.co/models?filter=fastspeech2
]


# Copied from transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextSinusoidalPositionalEmbedding with Speech2Text->FastSpeech2
class FastSpeech2PositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.size()
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        Args:
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx


class FastSpeech2SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

        self.kqv_proj = nn.Linear(3 * config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask):
        return F.multi_head_attention_forward(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            embed_dim_to_check=self.hidden_size,
            num_heads=self.num_attention_heads,
            in_proj_weight=self.kqv_proj.weight,
            add_zero_attn=False,
            dropout=self.attention_probs_dropout_prob,
            out_proj_weight=self.out_proj.weight,
            training=self.training,
            key_padding_mask=attention_mask,
        )


class FastSpeech2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = FastSpeech2SelfAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        hidden_states, attentions = self.self(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        return (hidden_states, attentions) if output_attentions else hidden_states


class FastSpeech2Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = 1 / math.sqrt(config.hidden_size)
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.intermediate_size,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        # hidden_states.shape == (seq_len, batch_size, num_channels)
        hidden_states = self.conv(hidden_states.permute(1, 2, 0)).permute(2, 0, 1)
        hidden_states = self.scale * hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class FastSpeech2Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dropout(self.dense(hidden_states))
        return hidden_states


class FastSpeech2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = FastSpeech2Attention(config)
        self.intermediate = FastSpeech2Intermediate(config)
        self.output = FastSpeech2Output(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        non_padding_mask = (1 - attention_mask.float()).transpose(0, 1)[..., None]

        residual = hidden_states * non_padding_mask
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = (residual + hidden_states) * non_padding_mask

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.output(self.intermediate(hidden_states))
        hidden_states = (residual + hidden_states) * non_padding_mask

        return hidden_states


class FastSpeech2Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([FastSpeech2Layer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.transpose(0, 1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        non_padding_mask = 1 - attention_mask.transpose(0, 1).float()[:, :, None]
        hidden_states = self.layer_norm(hidden_states) * non_padding_mask
        return hidden_states


class FastSpeech2Encoder(FastSpeech2Backbone):
    def __init__(self, config):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.phoneme_embed_scale = math.sqrt(config.hidden_size)
        self.phoneme_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.positional_embedding = FastSpeech2PositionalEmbedding(
            config.max_position_embeddings, config.hidden_size, config.pad_token_id
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        position_embeds = self.position_embedding(input_ids)
        phoneme_embeds = self.phoneme_embed_scale * self.phoneme_embedding(input_ids)
        embeds = self.dropout(phoneme_embeds + position_embeds)
        attention_mask = input_ids.eq(self.pad_token_id).detach()
        return super().forward(embeds, attention_mask)


class FastSpeech2Decoder(FastSpeech2Backbone):
    def __init__(self, config):
        super().__init__(config)
        self.positional_embedding_scale = nn.Parameter(torch.Tensor([1]))
        self.positional_embedding = FastSpeech2PositionalEmbedding(
            config.max_position_embeddings, config.hidden_size, config.pad_token_id
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mel_head = nn.Linear(config.hidden_size, config.mel_size)

    def forward(self, hidden_states, attention_mask):
        positional_embeds = self.positional_embedding_scale * self.positional_embedding(hidden_states[..., 0])
        hidden_states = self.dropout(hidden_states + positional_embeds)
        hidden_states = super().forward(hidden_states, attention_mask)
        mel = self.mel_head(hidden_states)
        return mel


class LayerNormTranspose(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(eps=1e-12, *args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.offset = 1.0
        kernel_size = config.duration_predictor_kernel_size
        padding = kernel_size // 2
        self.conv = nn.ModuleList()
        for _ in range(config.num_duration_predictor_layers):
            self.conv.append(
                nn.Sequential(
                    nn.ConstantPad1d((padding, padding), 0),
                    nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size),
                    nn.ReLU(),
                    LayerNormTranspose(config.hidden_size),
                    nn.Dropout(config.predictor_probs_dropout_prob),
                )
            )
        self.linear = nn.Linear(config.hidden_size, 1)

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]

        xs = self.linear(xs.transpose(1, -1))  # [B, T, C]
        xs = xs * (1 - x_masks.float())[:, :, None]  # (B, T, C)
        if is_inference:
            return self.out2dur(xs), xs
        return xs.squeeze(-1)  # (B, Tmax)

    def out2dur(self, xs):
        # NOTE: calculate in log domain
        xs = xs.squeeze(-1)  # (B, Tmax)
        dur = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()  # avoid negative value
        return dur

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return self._forward(xs, x_masks, True)


class LengthRegulator(nn.Module):
    def forward(self, dur, dur_padding=None, alpha=1.0):
        """
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0], [0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                         [0,0,2,2,0,0,0], [0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param dur_padding: Batch of padding of each frame (B, T_txt)
        :param alpha: duration rescale coefficient
        :return:
            mel2ph (B, T_speech)
        """
        assert alpha > 0
        device = dur.device
        dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(device)
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)
        pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(device)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask.long()).sum(1)
        return mel2ph


class CWTPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = config.predictor_kernel_size
        padding = kernel_size // 2
        self.linear = nn.Linear(config.hidden_size, config.cwt_hidden_size)
        # TODO: use `nn.Sequential` instead
        self.conv = nn.ModuleList()
        for idx in range(config.num_predictor_layers):
            in_channels = config.cwt_hidden_size if idx == 0 else config.hidden_size
            self.conv.append(
                nn.Sequential(
                    nn.ConstantPad1d((padding, padding), 0),
                    nn.Conv1d(in_channels, config.hidden_size, kernel_size),
                    nn.ReLU(),
                    LayerNormTranspose(config.hidden_size),
                    nn.Dropout(config.predictor_probs_dropout_prob),
                )
            )
        self.out = nn.Linear(config.hidden_size, config.cwt_out_size)
        self.positional_embedding = FastSpeech2PositionalEmbedding(4096, config.cwt_hidden_size, 0)
        self.positional_embedding_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs):
        """
        :param xs: [B, T, H] :return: [B, T, H]
        """
        xs = self.linear(xs)
        positions = self.positional_embedding_alpha * self.positional_embedding(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        # NOTE: calculate in log domain
        xs = self.out(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs


def get_cwt_stats_layers(config):
    return nn.Sequential(
        nn.Linear(config.hidden_size, config.cwt_hidden_size),
        nn.ReLU(),
        nn.Linear(config.cwt_hidden_size, config.cwt_hidden_size),
        nn.ReLU(),
        nn.Linear(config.cwt_hidden_size, 2),
    )


class FastSpeech2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FastSpeech2Config
    base_model_prefix = "fastspeech2"
    supports_gradient_checkpointing = True
    # TODO: is this the correct key(s)?
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            embedding_dim = module.embedding_dim
            module.weight.data.normal_(mean=0.0, std=embedding_dim ** -0.5)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FastSpeech2Encoder):
            module.gradient_checkpointing = value


FASTSPEECH2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~FastSpeech2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FASTSPEECH2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`FastSpeech2Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The FastSpeech2 Model that outputs predicted mel-spectrograms.",
    FASTSPEECH2_START_DOCSTRING,
)
class FastSpeech2Model(FastSpeech2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = FastSpeech2Encoder(config)
        self.decoder = FastSpeech2Decoder(config)
        self.length_regulator = LengthRegulator()
        self.duration_predictor = DurationPredictor(config)
        self.cwt_predictor = CWTPredictor(config)
        self.cwt_stats_layers = get_cwt_stats_layers(config)
        self.pitch_embedding = nn.Embedding(config.pitch_size, config.hidden_size, config.pad_token_id)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FASTSPEECH2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(self, input_ids, f0=None, uv=None):
        encoder_output = self.encoder(input_ids)
        input_non_padding_mask = (input_ids > 0).float()[:, :, None]

        # add duration
        duration_input = encoder_output * input_non_padding_mask
        mel2ph = self.add_dur(duration_input, input_ids)

        decoder_input = F.pad(encoder_output, [0, 0, 1, 0])

        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_output.shape[-1]])
        decoder_input_original = decoder_input = torch.gather(decoder_input, 1, mel2ph_)  # [B, T, H]

        output_non_padding_mask = (mel2ph > 0).float()[:, :, None]
        pitch_input = decoder_input_original * output_non_padding_mask

        pitch_input_ph = encoder_output * input_non_padding_mask
        decoder_input = decoder_input + self.add_pitch(pitch_input, f0, uv, mel2ph, encoder_out=pitch_input_ph)

    def add_duration(self, duration_input, input_ids):
        padding = input_ids == self.config.pad_token_id
        duration_input = duration_input.detach() + 0.1 * (duration_input - duration_input.detach())
        duration, _ = self.duration_predictor.inference(duration_input, padding)
        mel2ph = self.length_regulator(duration, padding).detach()
        return mel2ph

    def add_pitch(self, pitch_input, f0, uv, mel2ph, encoder_output):
        decoder_inp = pitch_input.detach() + hparams["predictor_grad"] * (pitch_input - pitch_input.detach())
        pitch_padding = mel2ph == 0
        pitch_padding = None
        ret["cwt"] = cwt_out = self.cwt_predictor(decoder_inp)
        stats_out = self.cwt_stats_layers(encoder_output[:, 0, :])  # [B, 2]
        mean = ret["f0_mean"] = stats_out[:, 0]
        std = ret["f0_std"] = stats_out[:, 1]
        cwt_spec = cwt_out[:, :, :10]
        if f0 is None:
            std = std * hparams["cwt_std_scale"]
            f0 = self.cwt2f0_norm(cwt_spec, mean, std, mel2ph)
            if hparams["use_uv"]:
                assert cwt_out.shape[-1] == 11
                uv = cwt_out[:, :, -1] > 0
        ret["f0_denorm"] = f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed
