from keras.datasets import fashion_mnist, mnist # type: ignore
import wandb

run = wandb.init(
    entity="da24m023-indian-institute-of-technology-madras",
    project="sample_images",
)

run.name = "sample_images"

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

sample_imgs = []

for i in range(10):
    for ind in range(x_train.shape[0]):
       if y_train[ind] == i:
          sample_imgs.append(wandb.Image(x_train[ind], caption=classes[i]))
          break
wandb.log({"sample_fashion_mnist_images": sample_imgs})

(x_train, y_train), (x_test, y_test) = mnist.load_data()
classes = [str(num) for num in range(10)]

sample_imgs = []

for i in range(10):
    for ind in range(x_train.shape[0]):
       if y_train[ind] == i:
          sample_imgs.append(wandb.Image(x_train[ind], caption=classes[i]))
          break
wandb.log({"sample_mnist_images": sample_imgs})

wandb.finish()