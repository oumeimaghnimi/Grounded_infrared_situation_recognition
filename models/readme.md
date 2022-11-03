from transformers import Pipeline

class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        #batching, transformation, indexing, dict_to_json,xml_to_json,txt_to_json,tokenization, input embedding
        model_input = Tensor(inputs["input_ids"])
        
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        
       '''May be
        
        class ViTModel(ViTPreTrainedModel):

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

 
       return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
         )
        '''
    
        return outputs

    def postprocess(self, model_outputs):
        
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
        
        '''
        Maybe attention features visualization
        attentions=encoder_outputs.attentions,
        get attentions of last layer
        attentions = outputs.attentions[-1] 
        nh = attentions.shape[1] # number of heads
  
        '''
        def get_attention_maps(pixel_values, attentions, nh):
        def visualize_attention(image):
        '''
