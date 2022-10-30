https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/pipelines#transformers.pipeline
https://huggingface.co/docs/transformers/v4.23.1/en/task_summary



Examples:


from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Sentiment analysis pipeline
pipeline("sentiment-analysis")

# Question answering pipeline, specifying the checkpoint identifier
pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased")

# Named entity recognition pipeline, passing in a specific model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
pipeline("ner", model=model, tokenizer=tokenizer)




class transformers.pipeline:
    The pipeline abstraction

       pipe = pipeline("text-classification")
       pipe("This restaurant is awesome")

    To call a pipeline on many items, you can either call with a list.
       pipe(["This restaurant is awesome", "This restaurant is aweful"])
    If you want to use a specific model from the hub you can ignore the task if the model on the hub already defines it:

        pipe = pipeline(model="roberta-large-mnli")
        pipe("This restaurant is awesome")

    To iterate of full datasets it is recommended to use a dataset directly.
      This means you don’t need to allocate the whole dataset at once, nor do you need to do batching yourself.
       This should work just as fast as custom loops on GPU. If it doesn’t don’t hesitate to create an issue.

       pipe = pipeline("text-classification")
       pipe(["This restaurant is awesome", "This restaurant is aweful"])


    1-     import datasets
           from transformers import pipeline
           from transformers.pipelines.pt_utils import KeyDataset
           from tqdm.auto import tqdm

           pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
           dataset = datasets.load_dataset("superb", name="asr", split="test")

           # KeyDataset (only *pt*) will simply return the item in the dict returned by the dataset item
           # as we're not interested in the *target* part of the dataset.
           for out in tqdm(pipe(KeyDataset(dataset, "file"))):
             print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....

    2-  For ease of use, a generator is also possible:
           def data():
              while True:
                       # This could come from a dataset, a database, a queue or HTTP request
                       # in a server
                       # Caveat: because this is iterative, you cannot use `num_workers > 1` variable
                       # to use multiple threads to preprocess data. You can still have 1 thread that
                       # does the preprocessing while the main runs the big inference
                    yield "This is a test"
           for out in pipe(data()):
               print(out)     

     See also:  

         Pipeline batching   

         Pipeline chunk batching 
             zero-shot-classification and question-answering are slightly specific in the sense,
              that a single input might yield multiple forward pass of a model. Under normal circumstances, 
              this would yield issues with batch_size argument.

          In order to circumvent this issue, both of these pipelines are a bit specific, they are ChunkPipeline instead of regular Pipeline. 
           
           In short:


          preprocessed = pipe.preprocess(inputs)
          model_outputs = pipe.forward(preprocessed)
          outputs = pipe.postprocess(model_outputs)

          Now becomes:

          all_model_outputs = []
           for preprocessed in pipe.preprocess(inputs):
              model_outputs = pipe.forward(preprocessed)
              all_model_outputs.append(model_outputs)
              outputs = pipe.postprocess(all_model_outputs)


           This should be very transparent to your code because the pipelines are used in the same way.

           This is a simplified view, since the pipeline can handle automatically the batch to ! 
           Meaning you don’t have to care about how many forward passes you inputs are actually going to trigger,
           you can optimize the batch_size independently of the inputs. The caveats from the previous section still apply.   



See parameters explication of :
class transformers.pipeline:

     -Model:                       example: AutoModelForTokenClassification,
     -Tokenizer:                   example: AutoTokenizer
     -use_auth_token (str or bool, optional) — The token to use as HTTP bearer authorization for remote files.
         If True, will use the token generated when running huggingface-cli login (stored in ~/.huggingface).
      ......
  Returns
    Pipeline
      A suitable pipeline for the task.


"""  Interessant note: 
     Utility factory method to build a Pipeline.

Pipelines are made of:

   A tokenizer in charge of mapping raw textual input to token.
   A model to make predictions from the inputs.
   Some (optional) post processing for enhancing model’s output.

"""





Example: 


from transformers import Pipeline

   https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/pipelines/base.py

         class Pipeline(_ScikitCompat):




"image-classification" pipeline is  in the following link: 
            https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/pipelines/image_classification.py#L32

                  class ImageClassificationPipeline(Pipeline):


Pipeline custom code:

     If you want to override a specific pipeline.

     Don’t hesitate to create an issue for your task at hand, the goal of the pipeline is to be easy to use and support most cases, so transformers could maybe support your use case.

     If you want to try simply you can:

     Subclass your pipeline of choice




class MyPipeline(TextClassificationPipeline):
    def postprocess():
        # Your code goes here
        scores = scores * 100
        # And here
        #....



my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
# or if you use *pipeline* function, then:
my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)

That should enable you to do all the custom code you want.



Implementing a new pipeline:

     https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/pipelines#transformers.pipeline
     https://huggingface.co/docs/transformers/v4.23.1/en/add_new_pipeline
     https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/pipelines/base.py


     PIPELINE_INIT_ARGS = r"""
    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        modelcard (`str` or [`ModelCard`], *optional*):
            Model card attributed to the model for this pipeline.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.
            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        task (`str`, defaults to `""`):
            A task-identifier for the pipeline.
        num_workers (`int`, *optional*, defaults to 8):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of
            workers to be used.
        batch_size (`int`, *optional*, defaults to 1):
            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of
            the batch to use, for inference this is not always beneficial, please read [Batching with
            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .
        args_parser ([`~pipelines.ArgumentHandler`], *optional*):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (`int`, *optional*, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
            the associated CUDA device id. You can pass native `torch.device` or a `str` too.
        binary_output (`bool`, *optional*, defaults to `False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
"""
             @add_end_docstrings(PIPELINE_INIT_ARGS)
             class Pipeline(_ScikitCompat):
                      def __init__(
                       model: 
                       tokenizer:
                       .....
                      ):







            The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across different pipelines.

Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following operations:

Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

Pipeline supports running on CPU or GPU through the device argument (see below).

Some pipeline, like for instance FeatureExtractionPipeline ('feature-extraction') output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we provide the binary_output constructor argument. If set to True, the output will be stored in the pickle format.          

     https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/pipelines/audio_classification.py#L66

     class AudioClassificationPipeline(Pipeline):
       
          """
    Audio classification pipeline using any `AutoModelForAudioClassification`. This pipeline predicts the class of a
    raw waveform or an audio file. In case of an audio file, ffmpeg should be installed to support multiple audio
    formats.
    This pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"audio-classification"`.
    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=audio-classification).
    """

    def __init__(self, *args, **kwargs):
        # Default, might be overriden by the model.config.
        # ( *args**kwargs ) sae as for Pipeline



https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/pipelines/feature_extraction.py
   class FeatureExtractionPipeline(Pipeline):

       



      Feature extraction pipeline using no model head. This pipeline extracts the hidden states from the base transformer, which can be used as features in downstream tasks.

      This feature extraction pipeline can currently be loaded from pipeline() using the task identifier: "feature-extraction".

     All models may be used for this pipeline. See a list of all models, including community-contributed models on huggingface.co/models.
         


Returns

     A nested list of float

     The features computed by the model.

     Extract the features of the input(s).



from transformers import Pipeline

class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class




class transformers.ImageSegmentationPipeline


   https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/pipelines#transformers.ImageSegmentationPipeline

  class transformers.ObjectDetectionPipeline
  class transformers.QuestionAnsweringPipeline
  class transformers.TextClassificationPipeline
  class transformers.VisualQuestionAnsweringPipeline
  class transformers.ZeroShotClassificationPipeline
  class transformers.ZeroShotObjectDetectionPipeline

see on https://huggingface.co/docs/transformers: section CONTRIBUTE, API, TUTORIALS, etc









https://huggingface.co/docs/transformers/v4.23.1/en/notebooks

