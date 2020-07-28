import argparse
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

parser = argparse.ArgumentParser(description='Train CIFAR-10 models.')
parser.add_argument('--init_para', default=0.0, type=float, help='initial parameters, 0.0: default')
parser.add_argument('--prune', default='0', type=int, help='compress level, 0: without pruning')
parser.add_argument('--train', default='1', type=int, help='compress level, 0: without training')

args = parser.parse_args()

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
if args.train: 
    if args.init_para == 0.0:
        initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    else:
        initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=None)


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

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

    model_path = 'model_' + str(args.init_para) + '_' + str(args.prune)
    model.save(model, model_path)
else:
    model_path = 'model_' + str(args.init_para) + '_' + str(args.prune)
    model = tf.keras.models.load_model(model_path)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_loss, test_acc)