import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def np_one_hot(x, num_classes, dtype=float):
  x = np.asarray(x)
  out = np.zeros(x.shape + (num_classes,), dtype=dtype)
  
  flat_x = x.ravel()
  out_flat = out.reshape(-1, num_classes)
  
  valid = (flat_x >= 0) & (flat_x < num_classes)
  out_flat[np.arange(flat_x.size)[valid], flat_x[valid]] = 1
  
  return out




def load_mnist(n_tr, n_te, seed, down=False, one_hot=False):
  ds = tfds.load("mnist", split="train", as_supervised=True)
  ds = ds.shuffle(buffer_size=60_000, seed=seed,reshuffle_each_iteration=False)
  images = []
  labels = []
  for img, lbl in tfds.as_numpy(ds.take(n_tr+n_te)):
    if down:
      img = tf.image.resize(img, (8, 8)).numpy()
    images.append(img.reshape(-1) / 255.0)
    labels.append(lbl)
  
  X = np.array(images)
  y = np.array(labels).astype(int)
  
  X_tr = X[:n_tr]
  X_te = X[n_tr:]
  y_tr = y[:n_tr]
  y_te = y[n_tr:]
  
  if one_hot:
    y_tr = np_one_hot(y_tr, 10)
    y_te = np_one_hot(y_te, 10)
  return X_tr, y_tr, X_te, y_te



@tf.autograph.experimental.do_not_convert
def load_cifar(n_tr, n_te, seed):
  ds = tfds.load("cifar10", split="train", as_supervised=True)
  ds = ds.filter(lambda img, lbl: (lbl == 3) | (lbl == 5))
  ds = ds.shuffle(buffer_size=10000, seed=seed, reshuffle_each_iteration=False)
    
  images = []
  labels = []
  for img, lbl in tfds.as_numpy(ds.take(n_tr + n_te)):
    img = tf.image.resize(img, (8, 8))
    img = tf.image.rgb_to_grayscale(img)
    images.append(img.numpy().reshape(-1))
    labels.append(lbl)
  
  X = np.array(images)
  y = (np.array(labels).reshape(-1,1) - 3) // 2
  
  X = (X-np.mean(X,0))/np.std(X,0)
  X_tr = X[:n_tr]
  X_te = X[n_tr:]
  y_tr = y[:n_tr]
  y_te = y[n_tr:]
  return X_tr, y_tr, X_te, y_te

