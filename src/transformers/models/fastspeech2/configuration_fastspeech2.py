# coding=utf-8
# Copyright 2022 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
""" FastSpeech2 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

FASTSPEECH2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/fastspeech2": "https://huggingface.co/microsoft/fastspeech2/resolve/main/config.json",
    # See all FastSpeech2 models at https://huggingface.co/models?filter=fastspeech2
}


class FastSpeech2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~FastSpeech2Model`]. It is used to instantiate an
    FastSpeech2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the FastSpeech2
    [microsoft/fastspeech2](https://huggingface.co/microsoft/fastspeech2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the FastSpeech2 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`~FastSpeech2Model`] or [`~TFFastSpeech2Model`].
        hidden_size (`int`, *optional*, defaults to 256):
            Dimension of the Transformer encoder layers.
        num_encoder_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers prior to the Length Regulator.
        num_decoder_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers after the Length Regulator.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        Example:

    ```python
    >>> from transformers import FastSpeech2Model, FastSpeech2Config

    >>> # Initializing a FastSpeech2 microsoft/fastspeech2 style configuration
    >>> configuration = FastSpeech2Config()

    >>> # Initializing a model from the microsoft/fastspeech2 style configuration
    >>> model = FastSpeech2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "fastspeech2"

    def __init__(
        self,
        vocab_size=79,
        hidden_size=256,
        num_hidden_layers=4,
        kernel_size=9,
        num_attention_heads=2,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        predictor_probs_dropout_prob=0.5,
        max_position_embeddings=2000,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        **kwargs
    ):
        if kernel_size % 2 == 0:
            raise ValueError(f"Expected `kernel_size` to be odd, but got {kernel_size}")
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.kernel_size = kernel_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.predictor_probs_dropout_prob = predictor_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        super().__init__(pad_token_id=pad_token_id, **kwargs)
