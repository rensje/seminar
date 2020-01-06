import tensorflow as tf
from util import log_normal_pdf


def encoder1(latent_dim, keep_prob):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding="same", activation=tf.nn.leaky_relu),
        tf.keras.layers.Dropout(keep_prob),
        tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding="same", activation=tf.nn.leaky_relu),
        tf.keras.layers.Dropout(keep_prob),
        tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=1, padding="same", activation=tf.nn.leaky_relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_dim + latent_dim)
        ])

def decoder1(latent_dim, keep_prob):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=latent_dim),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=1, padding="same", activation="relu"),
        tf.keras.layers.Dropout(keep_prob),
        tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same", activation="relu"),
        tf.keras.layers.Dropout(keep_prob),
        tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same", activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_dim + latent_dim)
    ])

def encoder2(latent_dim, keep_prob):
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

def decoder2(latent_dim, keep_prob):
    return tf.keras.Sequential(
            [
              tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
              tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
              tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
              tf.keras.layers.Conv2DTranspose(
                  filters=64,
                  kernel_size=3,
                  strides=(2, 2),
                  padding="SAME",
                  activation='relu'),
              tf.keras.layers.Conv2DTranspose(
                  filters=32,
                  kernel_size=3,
                  strides=(2, 2),
                  padding="SAME",
                  activation='relu'),
              # No activation
              tf.keras.layers.Conv2DTranspose(
                  filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

class VAE(tf.keras.Model):
  def __init__(self, latent_dim, inference_net = encoder2, generative_net = decoder2, optimizer = tf.keras.optimizers.Adam(1e-4), kl_importance=1, keep_prob=0.8):
    super(VAE, self).__init__()

    self.latent_dim = latent_dim
    self.kl_importance=kl_importance

    self.inference_net = inference_net(latent_dim, keep_prob)
    self.generative_net = generative_net(latent_dim, keep_prob)
    self.optimizer = optimizer

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=False)

  def encode(self, x, training=False):
    mean, logvar = tf.split(self.inference_net(x, training=training), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False, training=False):
    logits = self.generative_net(z, training=training)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits


  @tf.function
  def compute_loss(self, x, training=False):
      mean, logvar = self.encode(x, training=training)
      z = self.reparameterize(mean, logvar)
      x_pred = self.decode(z, training=training)

      rec_loss = tf.norm(tf.reshape(x, [x.shape[0], -1])-tf.reshape(x_pred, [x_pred.shape[0], -1]), axis=1)
      kl = tf.reduce_sum(mean ** 2 + tf.math.exp(logvar) - logvar - 1, axis=1)/2
      # kl = log_normal_pdf(z, mean, logvar)-log_normal_pdf(z, 0., 0.)

      c=self.kl_importance # some constant, can change later
      loss = tf.reduce_mean(rec_loss/(2*c) + kl)
      return loss



      # x_logit = self.decode(z)
      # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
      # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
      # logpz = log_normal_pdf(z, 0., 0.)
      # logqz_x = log_normal_pdf(z, mean, logvar)
      # return -tf.reduce_mean(logpx_z + logpz - logqz_x)





  @tf.function
  def compute_apply_gradients(self, x):
      with tf.GradientTape() as tape:
          loss = self.compute_loss(x, training=True)
      gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))