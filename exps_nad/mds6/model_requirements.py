"""
model_requirements defines the *necessary* structural and functional conditions that a model must satisfy for a particular predictive task.
These are not *sufficient* conditions for correctness or training success—they ensure interface compliance only.
Feel free to extend hyperparameters or functions.
For functions in the model:
 - Input signature: Each method takes one or more named tensor arguments, specified by parameter names and their shapes/types.
 - Output format: Each method returns a dictionary where each key corresponds to a named output tensor with a defined shape and type.
 - Ensure the methods include the specified arguments or keys, but they don't need to be limited to them. Adding extra and useful keys to the output dictionary is encouraged to support interaction between modules.
     - For example, you are free to output additional keys in the outputs dictionary.

Note: We use `predict` to output the predicted logit for evaluation. Do not use the ground truth label as the input of the `predict` function. We use `forward` to train the model. Use CPU for smoke testing.

Contract: `forward` and `predict` method must be tolerant of extra, unknown keyword arguments — implement the signature with **kwargs and silently ignore unrecognized keys.


"""
model_requirements = {
    "model_name": "Image3DClfModel",
    "purpose": "Predict labels of 3D medical images",
    "descriptive_requirement": "The input 'pixel_values' represents a rescaled-preprocessed 3D image. label_num is 11.",
    "init_parameters": {"label_num": {"type": "int"}, "base_dim": {"type": "int"}, "model_depth": {"type": "int"},},
    "methods": {
        "forward": {
            "inputs": {
                "pixel_values": {"shape": "(batch_size, 1, 64, 64, 64)", "type": "torch.FloatTensor"},
                "label": {"shape": "(batch_size)", "type": "torch.LongTensor"},
            },
            "outputs": {
                "logits": {"shape": "(batch_size, label_num)", "type": "torch.FloatTensor"},
                "loss": {"shape": "()", "type": "torch.FloatTensor"}
            }
        },
        "predict":{
            "inputs": {
                "pixel_values": {"shape": "(batch_size, 1, 64, 64, 64)", "type": "torch.FloatTensor"},
            },
            "outputs": {
                "logits": {"shape": "(batch_size, label_num)", "type": "torch.FloatTensor"},
            }
        }
    },
    "other_requirements": ["The `base_dim` defines a base dim of the model (default value is 32). Specific modules adjust this dimension by expanding or reducing it as needed. (base_dim*2 = 64, base_dim*4=128, base_dim*8=256, ...)",
                          "The `model_depth` defines the number of main layers or blocks in the model at a macro level — typically values like 12, 15 or 18. To create layers or stages, adapt them based on `model_depth`. E.g., use a safe splitting function or define stage_depth = ratio × model_depth.",
                        "Sample size is about 1k, small. Be cautious about overfitting and early saturation."],
}
