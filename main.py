import argparse
import tensorflow as tf
import copy
import numpy as np
import PIL

from tensorflow.keras import datasets, layers, models

parser = argparse.ArgumentParser(description='Train CIFAR-10 models.')
parser.add_argument('--init_para', default='zeros', type=str, help='initial parameters, 0.0: default')
parser.add_argument('--prune', default='0', type=int, help='compress level, 0: without pruning')
parser.add_argument('--train', default='1', type=int, help='compress level, 0: without training')

args = parser.parse_args()

def compute_distance(model1, model2):
    # model1_copy = copy.deepcopy(model1)
    # print(model1_copy)
    distance = 0.0
    for inx, layer1 in enumerate(model1.layers):
        layer2 = model2.layers[inx]
        weight_layer1 = layer1.get_weights()
        weight_layer2 = layer2.get_weights()
        if len(weight_layer1):
            distance += np.absolute(np.sum(np.power(np.subtract(layer1.get_weights()[0], layer2.get_weights()[0]), 2)))
            distance += np.absolute(np.sum(np.power(np.subtract(layer1.get_weights()[1], layer2.get_weights()[1]), 2)))
    distance = np.sqrt(distance)
    return distance
    

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
if args.train: 
    epochs = 10
    if args.init_para == 'zeros':
        initializer = tf.keras.initializers.Zeros()
        epochs = 1

    elif args.init_para == 'random_normal':
        initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    elif args.init_para == 'orthogonal':
        initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=None)
    else:
        print('No initializer')

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_initializer=initializer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_initializer=initializer))
    model.add(layers.Dense(10))

    model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    
    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(test_images, test_labels))

    model_path = 'model_' + args.init_para + '_' + str(args.prune)
    model.save(model_path)
else:
    model_path = 'model_' + args.init_para + '_' + str(args.prune)
    model = tf.keras.models.load_model(model_path)

distance = compute_distance(model, model)
print('distance :', distance)

# test on test dataset
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_loss, test_acc)

# test on individual images
black_image = np.array(PIL.Image.open('black.jpg'))
original_image = np.array(PIL.Image.open('original.jpg'))


black_output = model.predict(tf.expand_dims(black_image, 0))

print(black_output)

