
Build a great repository:
https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR

https://www.freecodecamp.org/news/how-to-deploy-your-machine-learning-model-as-a-web-app-using-gradio/
    
Example 1:
def greet_user(name):
	return "Hello " + name + " Welcome to Gradio!😎"

app =  gr.Interface(fn = greet_user, inputs="text", outputs="text")
app.launch()
    
Example 2:
def visualize_attention(image):
    return ..

title = "Interactive demo: DINO"
description = "Demo for Facebook AI's DINO, a new method for self-supervised training of Vision Transformers. Using this method, they are capable of segmenting objects within an image without having ever been trained to do so. This can be observed by displaying the self-attention of the heads from the last layer for the [CLS] token query. This demo uses a ViT-S/8 trained with DINO. To use it, simply upload an image or use the example image below. Results will show up in a few seconds."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.14294'>Emerging Properties in Self-Supervised Vision Transformers</a> | <a href='https://github.com/facebookresearch/dino'>Github Repo</a></p>"
examples =[['cats.jpg']]

iface = gr.Interface(fn=visualize_attention, 
                     inputs=gr.inputs.Image(shape=(480, 480), type="pil"), 
                     outputs=[gr.outputs.Image(type='file', label=f'attention_head_{i}') for i in range(6)],
                     title=title,
                     description=description,
                     article=article,
                     examples=examples)
iface.launch()

https://huggingface.co/docs

https://github.com/huggingface/datasets
https://github.com/huggingface/datasets/tree/main/src/datasets

https://github.com/huggingface/datasets/blob/main/src/datasets/load.py
      def load_dataset()


The bare ViT Model transformer outputting raw hidden-states without any specific head on top.(class transformers.ViTModel, class transformers.TFViTModel, class transformers.FlaxViTModel)

for class transformers.ViTForImageClassification or class transformers.FlaxViTForImageClassification: ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the [CLS] token) e.g. for ImageNet.
for class transformers.ViTForMaskedImageModeling: ViT Model with a decoder on top for masked image modeling, as proposed in SimMIM.
https://huggingface.co/facebook/dino-vits8/tree/main

https://huggingface.co/spaces/nielsr/DINO/blob/main/app.py

The above model /app is contributed by nielsr https://huggingface.co/nielsr
           The original code (written in JAX) can be found   : https://github.com/google-research/vision_transformer

Note that In huggingface/viT  the weights is converted from Ross Wightman’s timm library,(https://github.com/rwightman/pytorch-image-models) 
    who already converted the weights from JAX to PyTorch. 

pixel_values (torch.FloatTensor of shape (batch_size, num_channels, height, width))
last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.

pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

""" https://theaisummer.com/ (very good)
https://huggingface.co/docs/transformers/model_doc/vit

https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer

https://theaisummer.com/hugging-face-vit/:  A complete Hugging Face tutorial: how to build and train a vision transformer


https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/feature_extraction_vit.py """




import os
os.system('pip install git+https://github.com/huggingface/transformers.git --upgrade')

from transformers import ViTFeatureExtractor, ViTModel

import torch
from PIL import Image
import requests



#https://github.com/facebookresearch/dino
# https://huggingface.co/facebook/dino-vits8









#I- image dowlading:
#torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')
#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)

#from datasets import load_dataset

#dataset = load_dataset("huggingface/cats-image")
#image = dataset["test"]["image"][0]








#II-Model configuration:


       #from  configuration_utils import PretrainedConfig
          # https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py
          


       #https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/configuration_vit.py    
         #VIT_PRETRAINED_CONFIG_ARCHIVE_MAP = { "google/vit-base-patch16-224": "https://huggingface.co/vit-base-patch16-224/resolve/main/config.json",
         # See all ViT models at https://huggingface.co/models?filter=vit}

      # class ViTConfig(PretrainedConfig):

           #1- model.config.patch_size:



#III-Model building:                                ##from .models.vit import ViTModel
#https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py


    #class ViTEmbeddings(nn.Module):
    #class ViTSelfAttention(nn.Module):
    #class ViTSelfOutput(nn.Module):
    #class ViTAttention(nn.Module):
    #class ViTIntermediate(nn.Module):
    #class ViTOutput(nn.Module):
    #class ViTLayer(nn.Module):
    #class ViTEncoder(nn.Module):
    #class ViTPooler(nn.Module):
    #class ViTPatchEmbeddings(nn.Module):     
'''  ViTPatchEmbeddings turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `    hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
         Transformer.


    '''


    #from ...configuration_utils import PretrainedConfig: https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py
    
'''configuration common to all models
      Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
      methods for loading/downloading/saving configurations.

      class PretrainedConfig(PushToHubMixin):
        """
        PushToHubMixin: https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py
                        Hub utilities: utilities related to download and cache models
                        huggingface_hub: import huggingface_hub
        """

    '''
     

    
    # ViTConfig:  (Model configuration) in case of ViT model
            # from .configuration_vit import ViTConfig: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/configuration_vit.py
    


    #from ...configuration_utils import PretrainedConfig:
    #class ViTConfig(PretrainedConfig):
'''
    This is the configuration class to store the configuration of a [`ViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to `224`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to `16`):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to `3`):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        encoder_stride (`int`, `optional`, defaults to 16):
           Factor to increase the spatial resolution by in the decoder head for masked image modeling.
    
        
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ViT google/vit-base-patch16-224 architecture.
    Example:


    ```python

    >>> from transformers import ViTModel, ViTConfig
    >>> # Initializing a ViT vit-base-patch16-224 style configuration
    >>> configuration = ViTConfig()
    >>> # Initializing a model from the vit-base-patch16-224 style configuration
    >>> model = ViTModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
'''


#class ViTPreTrainedModel(PreTrainedModel):
#class ViTModel(ViTPreTrainedModel):
    
# PreTrainedModel:
   # from ...modeling_utils import PreTrainedModel: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py

'''
        Base class for all models.
       [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
        downloading and saving models as well as a few methods common to all models to:
        - resize the input embeddings,
        - prune heads in the self-attention heads.
     
     
     '''

'''
       **config_class** ([`PretrainedConfig`]) 
      **load_tf_weights** (`Callable`):
           - **model** ([`PreTrainedModel`])
           - **config** ([`PreTrainedConfig`])
           - **path** (`str`)
       **base_model_prefix** (`str`)
       **is_parallelizable** (`bool`)
       **main_input_name** (`str`) 
      '''

"""An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
        models.
               config_class = ViTConfig
               base_model_prefix = "vit"
               main_input_name = "pixel_values"
               supports_gradient_checkpointing = True
        
"""   

"""   https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py


#.from_pretrained

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        
        Instantiate a pretrained pytorch model from a pre-trained model configuration.
        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.
        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.
        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

                                                          ~.from_pretrained(  ,*model_args, **kwargs)

[`~PretrainedConfig.from_pretrained`]:
         cls.config_class.from_pretrained      #cls.config_class = ViTConfig.from_pretrained
[`~PreTrainedModel.from_pretrained`]           ViTModel(ViTPreTrainedModel) #ViTPreTrainedModel(PreTrainedModel)
         ViTModel.from_pretrained         
ViTFeatureExtractor.from_pretrained

       Examples:
        ```python
        >>> from transformers import BertConfig, BertModel
        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
        >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
        >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
        >>> model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)
        ```
"""





    #class ViTPreTrainedModel(PreTrainedModel): https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py


     

    #class ViTForImageClassification(ViTPreTrainedModel): https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
    #Masked Prediction: class ViTForMaskedImageModeling(ViTPreTrainedModel): https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
   
   
    #class ViTModel(ViTPreTrainedModel): https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py



"""class ViTModel(ViTPreTrainedModel):
         def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
         super().__init__(config)
          self.config = config
        """



'''def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):'''
'''*model_args  :    self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False


All checkpoints can be found on the hub: for ViT:    https://huggingface.co/models?search=vit, for Dino: https://huggingface.co/models?other=dino
'''
#ViTModel.from_pretrained("facebook/dino-vits8", config:ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False)
#from transformers import ViTModel


model = ViTModel.from_pretrained("facebook/dino-vits8", add_pooling_layer=False)




"""       class ViTModel(ViTPreTrainedModel) in detail



VIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
'''
           ViTModel_step_1 # Enter the configuration parameters as arguments to ---->ViTConfig
                           # We obtain ViTConfig     **Note: ViTConfig can be obtained from pretrained model: ViTConfig.from_pretrained ()#see def from_pretrained
                           # idea: we can do hyperparameter optimization


           ViTModel_step_2 # feature extraction     to have batched pixel_values: Optional[torch.Tensor]  
                 Main method to prepare for the model one or several image(s).
           As the Vision Transformer expects each image to be of the same size (resolution), 
               one can use ViTFeatureExtractor to resize (or rescale) and normalize images for the model.

                             #from .models.vit import ViTFeatureExtractor
                                feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/dino-vits8", do_resize=False)
                            see   [`ViTFeatureExtractor.__call__`] for details

               See also class transformers.ImageFeatureExtractionMixin
               https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/image_utils.py
                  def center_crop
                  def convert_rgb
                  def expand_dims
                  def flip_channel_order
                  def normalize
                  def rescale
                  def resize
                  def rotate
                  def to_numpy_array
                  def to_pil_image

           ViTModel_step_3 # prepare  VIT_INPUTS_DOCSTRING

                          *pixel_values: Optional[torch.Tensor] 
                                              pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                                              Pixel values. Pixel values can be obtained using [`ViTFeatureExtractor`]. See
                                              [`ViTFeatureExtractor.__call__`] for details.

                          *bool_masked_pos: Optional[torch.BoolTensor] 
                          *head_mask: Optional[torch.Tensor] 
                          *output_attentions: Optional[bool] 
                          *output_hidden_states: Optional[bool] 
                          *interpolate_pos_encoding: Optional[bool] 
                               [The Vision Transformer was pre-trained using a resolution of 224x224. During fine-tuning, 
                                it is often beneficial to use a higher resolution than pre-training (Touvron et al., 2019), (Kolesnikov et al., 2020).
                                In order to fine-tune at higher resolution, the authors perform 2D interpolation of the pre-trained position embeddings,
                                  according to their location in the original image.]
                          *return_dict: Optional[bool] 
   
            ViTModel_step_4 # indicate boolean values of the followong entries
              add_pooling_layer: bool = True, 
              use_mask_token: bool = False

           ViTModel_step_3 # Build ViT architecture and entering VIT_INPUTS_DOCSTRING

                    ViTEmbeddings
                    ViTEncoder
                    nn.LayerNorm
                    ViTPooler      if add_pooling_layer else None

                    ViTPatchEmbeddings

                    for layer, heads in heads_to_prune.items():
                       self.encoder.layer[layer].attention.prune_heads(heads)

           ViTModel_step_4 # See https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
                   

                    # Initialize weights and apply final processing
                       
                             post_init(), .from_pretrained, 

                    # PreTrainedModel:

                              - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
                                        for this model architecture.
                              - **load_tf_weights** (`Callable`) -- A python *method* for loading a TensorFlow checkpoint in a PyTorch model,
                               taking as arguments:
                                  - **model** ([`PreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
                                  - **config** ([`PreTrainedConfig`]) -- An instance of the configuration associated to the model.
                                  - **path** (`str`) -- A path to the TensorFlow checkpoint.
                              - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
                              classes of the same architecture adding modules on top of the base model.
                              - **is_parallelizable** (`bool`) -- A flag indicating whether this model supports model parallelization.
                              - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
                                   models, `pixel_values` for vision models and `input_values` for speech models).
    """
                          config_class = None
                          base_model_prefix = ""
                          main_input_name = "pixel_values"      #
                          _auto_class = None
                          _no_split_modules = None
                         _keys_to_ignore_on_load_missing = None
                         _keys_to_ignore_on_load_unexpected = None
                         _keys_to_ignore_on_save = None
                         is_parallelizable = False
                         supports_gradient_checkpointing = False

                                   Example: config_class = ViTConfig
                                            base_model_prefix = "vit"
                                            main_input_name = "pixel_values"
                                            supports_gradient_checkpointing = True
  


                         @add_code_sample_docstrings(
                         processor_class=_FEAT_EXTRACTOR_FOR_DOC,
                         checkpoint=_CHECKPOINT_FOR_DOC,
                         output_type=BaseModelOutputWithPooling,
                         config_class=_CONFIG_FOR_DOC,
                         modality="vision",
                        expected_output=_EXPECTED_OUTPUT_SHAPE,
                        )        


           ViTModel_step_5 # The outputs of ViTModel  # transformers/modeling_outputs.py # class BaseModelOutputWithPooling(ModelOutput)

                        return BaseModelOutputWithPooling(
                              last_hidden_state=sequence_output,
                              pooler_output=pooled_output,
                             hidden_states=encoder_outputs.hidden_states,
                          attentions=encoder_outputs.attentions, )




                   
      class BaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
       

###

    class ViTConfig(PretrainedConfig):
    model_type = "vit"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        **kwargs):

        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride

'''
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information: https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/configuration#transformers.PretrainedConfig
    See sections  API :Configuration

    Initializing with a config file does not load the weights associated with the model, only the
    configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
class transformers.ViTModel  --->    Returns  transformers.modeling_outputs.BaseModelOutputWithPooling or tuple(torch.FloatTensor)


  in def forward()
VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViTFeatureExtractor`]. See
            [`ViTFeatureExtractor.__call__`] for details.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.




"""


@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)

class ViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
       
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


"""


#IV- feature extracting                               
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/feature_extraction_vit.py
   #from .models.vit import ViTFeatureExtractor

IV-1- class ViTFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a ViT feature extractor.
    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    ImageFeatureExtractionMixin: Mixin that contain utilities for preparing image features.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int` or `Tuple(int)`, *optional*, defaults to 224):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then the input will be resized to (size, size). Only has an effect if `do_resize` is
            set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`List[int]`, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """


 iV-2- def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
   args of ViTFeatureExtractor:  do_resize, size,  resample, do_normalize, image_mean, image_std


 ViTFeatureExtractor

    def __init__(
        self,
        do_resize=True,
        size=224,
        resample=Image.BILINEAR,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD



feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/dino-vits8", do_resize=False)



IV-3- Utilisation for given inputs:

  __call__ method of  ViTFeatureExtractor:

    def __call__(
        self, images: ImageInput,   return_tensors: Optional[Union[str, TensorType]] = None, **kwargs
    ) -> BatchFeature:
        """

        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>
        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor.
                 In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                 number of channels, H and W are image height and width.


            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
               -`'np'`: Return NumPy `np.ndarray` objects.
             -'jax'`: Return JAX `jnp.ndarray` objects.

            tensor_type (`str` or [`~utils.TensorType`], *optional*):  The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]     /transformers/utils/generic.py

             PYTORCH = "pt"
            TENSORFLOW = "tf"
            NUMPY = "np"
          JAX = "jax"

          see in huggingface page:  General Utilities:  class transformers.TensorType
"""

IV-4- : construct  UserDict containing the data after checking its type, type must be:
  `PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor],  
   normalizing the data , resizing it

   # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:                         # make the images in a list 
            images = [images]

        # transformations (resizing + normalization)
        if self.do_resize and self.size is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

     ----> data = {"pixel_values": images}

IV-5-   Do batching:  /transformers/feature_extraction_utils.py

"""
class BatchFeature(UserDict):   
    in huggingface page main_classes/feature_extractor#transformers.BatchFeature
    
r"""
     Holds the output of the pad()  [`~SequenceFeatureExtractor.pad`] and feature extractor specific `__call__` methods.
                               class transformers.SequenceFeatureExtractor
                              /main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad
      

    This class is derived from a python dictionary and can be used as a dictionary.
    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
            etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """       .
         .
     .


        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).

         For ViTFeatureExtractor:
              # return as BatchFeature
                 data = {"pixel_values": images}
                 encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

                 return encoded_inputs

"""    


IV-5-1:    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):       #feature_extraction_utils.py
              """
               Convert the inner content to tensors.
                 Args:
                       tensor_type (`str` or [`~utils.TensorType`], *optional*):
                     The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If None`, no modification is done.
        """

          as_tensor = np.asarray/
            is_tensor = _is_numpy/

            as_tensor = tf.constant
            is_tensor = tf.is_tensor

            as_tensor = jnp.array
            is_tensor = _is_jax


            as_tensor = np.asarray
            is_tensor = _is_numpy
.
.
.
.# Do the tensor conversion in batch
for key, value in self.items():

            try:

                if not is_tensor(value):

                    tensor = as_tensor(value)

                    self[key] = tensor

            except:  # noqa E722

                if key == "overflowing_values":

                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")

                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        return self

IV-5-2: Utilities to makes input (data) into dict      k:v, into self, etc

def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def __getitem__(self, item: str) -> Union[Any]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_values', 'attention_mask',
        etc.).
        """
        if isinstance(item, str):
            return self.data[item]
        else:
            raise KeyError("Indexing with integers is not available when using Python based feature extractors")

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

    # Copied from transformers.tokenization_utils_base.BatchEncoding.keys
    def keys(self):
        return self.data.keys()

    # Copied from transformers.tokenization_utils_base.BatchEncoding.values
    def values(self):
        return self.data.values()

    # Copied from transformers.tokenization_utils_base.BatchEncoding.items
    def items(self):
        return self.data.items()
    

IV-5-3- move to device the values of data:
    @torch_required
    # Copied from transformers.tokenization_utils_base.BatchEncoding.to with BatchEncoding->BatchFeature
    def to(self, device: Union[str, "torch.device"]) -> "BatchFeature":
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).
        Args:
            device (`str` or `torch.device`): The device to put the tensors on.
        Returns:
            [`BatchFeature`]: The same instance after modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            logger.warning(f"Attempting to cast a BatchFeature to type {str(device)}. This is not supported.")
        return self
"""
IV-6:     For more utilities in feature extraction: 
       The feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users   should refer to this superclass for more information regarding those methods.#transformers/feature_extraction_utils.py
                    

                                class FeatureExtractionMixin(PushToHubMixin):
                                           """
                                    This is a feature extraction mixin used to provide saving/loading functionality for sequential and image feature extractors.
                                            from .hub import  PushToHubMixin
                                           [`~utils.PushToHubMixin.push_to_hub`]

                                         """
                                                     1          @classmethod
                                                               def from_pretrained(  cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs ) -> PreTrainedFeatureExtractor:
                                                                     r"""
                                                                            Instantiate a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a feature extractor, *e.g.* a
                                                                              derived class of [`SequenceFeatureExtractor`].
                                                                                 """
                                                      2          def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
                                                                              """
                                                                               Save a feature_extractor object to the directory `save_directory`, so that it can be re-loaded using the
                                                                                 [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] class method.
                                                                                      """
                                                     3         @classmethod
                                                                    def get_feature_extractor_dict( cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                                                                                   """
                                                                                       From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
                                                                                            feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] using `from_dict`."""
                                                  4             @classmethod
                                                                    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> PreTrainedFeatureExtractor:
                                                                                """
                                                                                       Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of  parameters.
                                                                                       """
                                                  5           @classmethod
                                                                    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> PreTrainedFeatureExtractor:
                                                                                    """
                                                                                             Instantiates a feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] from the path to  a JSON file of parameters.  """
                                                  6           def to_json_string(self) -> str:
                                                                                  """
                                                                          Serializes this instance to a JSON string.
                                                                                      Returns:
                                                                                        `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
                                                                                         """



                                               7           def to_json_file(self, json_file_path: Union[str, os.PathLike]):
                                                                         """
                                                                               Save this instance to a JSON file.    

                                             8           @classmethod
                                                                  def register_for_auto_class(cls, auto_class="AutoFeatureExtractor"):
                                                                          """
                                                                           Register this class with a given auto class. This should only be used for custom feature extractors as the ones  in the library are already mapped with `AutoFeatureExtractor`.  """

                                        FeatureExtractionMixin.push_to_hub = copy_func(FeatureExtractionMixin.push_to_hub)
                                        FeatureExtractionMixin.push_to_hub.__doc__ = FeatureExtractionMixin.push_to_hub.__doc__.format(  object="feature extractor", object_class="AutoFeatureExtractor", object_files="feature extractor file")








# Prediction on unseen data for classification
    
#  class ViTEncoder(nn.Module):
#class ViTForImageClassification(ViTPreTrainedModel):
'''inputs = feature_extractor(images=image, return_tensors="pt")
   outputs = model(**inputs)
   last_hidden_states = outputs.last_hidden_state

'''


#Prediction on unseen data for Segmentation task



pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

  # forward pass
outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)


# attentions=output_attentions = encoder_outputs.attentions,


'''
  1-class ViTModel(ViTPreTrainedModel):

     def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):  


    .....

        return BaseModelOutputWithPooling(
                     last_hidden_state=sequence_output,
                     pooler_output=pooled_output,
                     hidden_states=encoder_outputs.hidden_states,
                     attentions=encoder_outputs.attentions,
        )

   
   """  BaseModelOutputWithPooling ??

from .utils import ModelOutput

@dataclass
 transformers.modeling_outputs.BaseModelOutputWithPooling
class BaseModelOutputWithPooling(ModelOutput):    #/transformers/modeling_outputs.py

    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

"""
   """



  2-        
      self.encoder = ViTEncoder(config)

      encoder_outputs = self.encoder(
                        embedding_output,
                        head_mask=head_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        )


  3-  class ViTEncoder(nn.Module):


     def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:


     
      if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

          return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
'''











#with torch.no_grad():
# forward pass
    outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)

#last_hidden_states = outputs.last_hidden_state
#list(last_hidden_states.shape)
#[1, 197, 768]

# get attentions of last layer
   attentions = outputs.attentions[-1] 
   nh = attentions.shape[1] # number of heads

#attentions (tuple(torch.FloatTensor), optional,
#returned when output_attentions=True is passed or when config.output_attentions=True) — 
#Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).

# we keep only the output patch attention
attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

#TORCH.SORT: Sorts the elements of the input tensor along a given dimension in ascending order by value.
#torch.sort(input, dim=- 1, descending=False, stable=False, *, out=None) #default: descending=False
#>>> x = torch.randn(3, 4)
>>> sorted, indices = torch.sort(x)
>>> sorted
tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
        [-0.5793,  0.0061,  0.6058,  0.9497],
        [-0.5071,  0.3343,  0.9553,  1.0960]])
>>> indices
tensor([[ 1,  0,  2,  3],
        [ 3,  1,  0,  2],
        [ 0,  3,  1,  2]])

#torch.sum(val, dim=1, keepdim=True)   
# normalize over sum
# val /= torch.sum(val, dim=1, keepdim=True)     
#torch.cumsum()
#Returns the cumulative sum of elements of input in the dimension dim.

    #For example, if input is a vector of size N, the result will also be a vector of size N, with elements.

        #y_i = x_1 + x_2 + x_3 + \dots + x_i


#torch.argsort(input, dim=- 1, descending=False) → LongTensor
#Returns the indices that sort a tensor along a given dimension in ascending order by value.

#This is the second value returned by torch.sort()
  input (Tensor) – the input tensor.

    dim (int, optional) – the dimension to sort along

    descending (bool, optional) – controls the sorting order (ascending or descending)''''''
#
#>>> a = torch.randn(4, 4)
#>>> a
#tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
        [ 0.1598,  0.0788, -0.0745, -1.2700],
        [ 1.2208,  1.0722, -0.7064,  1.2564],
        [ 0.0669, -0.2318, -0.8229, -0.9280]])


#>>> torch.argsort(a, dim=1)
#tensor([[2, 0, 3, 1],
        [3, 2, 1, 0],
        [2, 1, 0, 3],
        [3, 2, 1, 0]])

#Tensor.detach()
#Returns a new Tensor, detached from the current graph.
#The result will never require gradient. This method also affects forward mode AD gradients and the result will never have forward mode AD gradients.

#you can visualize  full attentions (attentions) or keeping only a certain percentage of the mass(th_attn)
def get_attention_maps(pixel_values, attentions, nh):
  threshold = 0.6
  w_featmap = pixel_values.shape[-2] // model.config.patch_size
  h_featmap = pixel_values.shape[-1] // model.config.patch_size

  # we keep only a certain percentage of the mass
  val, idx = torch.sort(attentions)
  val /= torch.sum(val, dim=1, keepdim=True)
  cumval = torch.cumsum(val, dim=1)
  th_attn = cumval > (1 - threshold)
  idx2 = torch.argsort(idx)
  for head in range(nh):
      th_attn[head] = th_attn[head][idx2[head]]
  th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
  # interpolate
  th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu().numpy()

  attentions = attentions.reshape(nh, w_featmap, h_featmap)
  attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu()
  attentions = attentions.detach().numpy()

  # save attentions heatmaps and return list of filenames
  output_dir = '.'
  os.makedirs(output_dir, exist_ok=True)
  attention_maps = []
  for j in range(nh):
      fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
      # save the attention map
      plt.imsave(fname=fname, arr=attentions[j], format='png')
      # append file name
      attention_maps.append(fname)

  return attention_maps
















attention_maps = get_attention_maps(pixel_values, attentions, nh)


        













# Feature map visualization:

def visualize_attention(image):
  # normalize channels
  pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values 

  # forward pass
  outputs = model(pixel_values, output_attentions=True, interpolate_pos_encoding=True)

  # get attentions of last layer
  attentions = outputs.attentions[-1] 
  nh = attentions.shape[1] # number of heads

  # we keep only the output patch attention
  attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

  attention_maps = get_attention_maps(pixel_values, attentions, nh)
  
  return attention_maps