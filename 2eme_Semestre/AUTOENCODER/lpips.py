import tensorflow.keras.models as KM
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
import tensorflow as tf

def tensor_normalizer(input: tf.Tensor, eps: float = 1e-10) -> tf.Tensor:
    norm_factor = K.sqrt(K.sum(K.square(input), axis=-1, keepdims=True))
    return input / (norm_factor + eps)

def perceptual_layer(input_tensor_shape: tuple, weights: float = 0.006, drop: float = 0.5) -> KM.Model:
    """
    Learned Perceptual Image Patch Similarity (LPIPS) distance.
    As proposed in:
        Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
        In CVPR, 2018.
    Code reference: https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
    
    The implementation is using resnet-50 feature extractor as reference

    Args:
        weights: weighing for the loss.
        drop: randomly drop channels for better regularization.

    Returns:
        KM.Model: LPIPS calculating model
    """
    resnet50 = KM.models.ResNet50(include_top=False, weights="imagenet", input_shape=input_tensor_shape)
    # Concatenated grund truth (GT) images and corresponding output images
    input_tensor = KL.Input(shape=input_tensor_shape) # Specify the tensor shape
    
    # Layers from which latent space features are to be extracted
    feature_names = [ "block_group1", "block_group2", "block_group3", "block_group4"]
    
    # Extract the latent features
    outputs = [resnet50.get_layer(feature).output for feature in feature_names]
    
    # Prepare the multilayer latent feature extractor model
    features_extractor = KM.Model(inputs=resnet50.input, 
                                  outputs=outputs)
    
    # extractor is trained with (0, 1) normalized images
    feature_maps = feature_extractor(input_tensor)
    
    error = 0.0 # initialize
    for i, feature in enumerate(feature_maps):
        # Extract deep features for the GT and output images
        content, generated = tf.split(feature, num_or_size_splits=2, axis=0)
        content, generated = tensor_normalizer(content), tensor_normalizer(generated)

        # Randomly zero-out some feature differences
        drop_features = KL.SpatialDropout2D(rate=drop)(K.square(content - generated))
        weighted_features = KL.Conv2D(
            1,
            1,
            (1, 1),
            name=f"lpips_{i+1}",
            kernel_initializer="he_normal",
            use_bias=False,
        )(drop_features)
        error = error + K.mean(weighted_features, axis=[-3, -2, -1])

    return KM.Model(
        inputs=input_tensor,
        outputs=(weights / (1 - drop)) * error,
        name="lpips_loss",
    )