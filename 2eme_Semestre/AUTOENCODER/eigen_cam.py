# ce code permet est une implémentation de la méthode eigen_cam permettant de récupérer des cartes de saillance. 
# elles sont basées sur la décomposition de valeur de singulière. En gros on projette la sortie de la toute dernière couche de convolution sur le premier vecteur singulier droit de l'équation. C'est la méthode avec laquelle j'ai récupéré les cartes qui sont dans ce dossier. 
# j'ai dû choisir cette méthode moins par choix que par praticité, en effet c'est une des rares à ne pas trop imposer de contraintes sur la couche de sortie du réseau et donc la tâche effectuée. Puisque la couche de sortie de notre tâche d'identification est assez particulière et qu'elle n'effectue pas une classification, c'était en fait la première méthode disponible. Donc, si on vous dit pendant votre soutenance que cette méthode n'est pas adaptée (la SVD supposant la linéarité qui est une hypothèse baffouée en réseaux de neurones), vous pouvez répondre ça.  
import tensorflow as tf
def eigen_cam(model, img, layer_name = None, label_index=None):
    # pour toute entrée, lui rajoute une dimension batch first si pas déjà présente:
    img = tf.convert_to_tensor(img) if not tf.is_tensor(img) else img
    img = tf.expand_dims(img, axis=0) if len(img.shape) == 3 else img # on travaille donc ici sur du (1, 224, 224, 3), càd du (batch, hauteur, largeur, canal)
    img_height, img_width = img.shape[1], img.shape[2]
    # extract the label_index of the last conv layer : 
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
        if layer_name is None:
            raise ValueError("Aucune couche de convolution trouvée dans le modèle.")

    layer = model.get_layer(layer_name).output
    activation_model = tf.keras.Model(inputs = model.input, outputs = [model.output, layer])
    
    # 1°) Forward pass pour obtenir preds et activations de la layer
    preds, feature = activation_model(img)

    # 2°) décomposition en valeur singulière
    s, u, v = tf.linalg.svd(feature, full_matrices = True)
    vT = tf.transpose(v, [0, 1, 3, 2])

    # 3°) Calcul de la carte CAM : 
    # On multiplie d'abord s par vT : s est scalaire pour chaque composante singulière, cela revient à une multiplication element-wise
    scaled_vT = s[..., 0, None, None] * vT[..., 0, None, :]
    # Ensuite, on fait une unique multiplication matricielle avec u
    eigen_cam = tf.linalg.matmul(u[..., 0, None], scaled_vT)

    # 4°) On somme sur l'axe canal 
    eigen_cam = tf.reduce_sum(eigen_cam, axis = -1, keepdims = True)

    # 5°) Resize à l'espace (i,j) de l'image et normalisation
    eigen_cam = tf.image.resize(eigen_cam, (img_height, img_width))
    eigen_cam_min, eigen_cam_max = tf.reduce_min(eigen_cam), tf.reduce_max(eigen_cam)
    eigen_cam = (eigen_cam - eigen_cam_min) / (eigen_cam_max - eigen_cam_min)
   
    eigen_cam = tf.squeeze(eigen_cam, axis = 0)
    return eigen_cam