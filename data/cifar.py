import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

IMG_MEAN = (0.5, 0.5, 0.5)
IMG_STD = (0.2, 0.2, 0.2)

def load_dataset(batch_dims, train=True, shuffle=True, repeat=True):
    total_batch_size = np.prod(batch_dims)
    
    if train:
        ds = tfds.load(
            f'cifar{NUM_CLASSES}',
            data_dir='../../tensorflow_datasets/',
            split='train',
        )
    else:
        ds = tfds.load(
            f'cifar{NUM_CLASSES}',
            data_dir='../../tensorflow_datasets/',
            split='test',
        )
        
    def preprocess(sample):
        img = tf.cast(sample['image'], tf.float32) / 255.0
        img -= tf.constant(IMG_MEAN, shape=(1,1,3))
        img /= tf.constant(IMG_STD, shape=(1,1,3))
        label = tf.one_hot(sample['label'], NUM_CLASSES)
        return {'x':img, 'y':label}
    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(NUM_TRAIN)
    if repeat:
        ds = ds.repeat()
        
    def aug(sample):
        img = sample['x']
        img = tf.image.random_flip_left_right(img)
        img = tf.pad(img, [[4,4], [4,4], [0,0]], mode='REFLECT')
        img = tf.image.random_crop(img, (32, 32, 3))
        return {'x':img, 'y':sample['y']}
    
    if train:
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.batch(total_batch_size, drop_remainder=True)
    
    def batch_reshape(batch):
        for k,v in batch.items():
            batch[k] = tf.reshape(v, batch_dims+v.shape[1:])
        return batch
    
    ds = ds.map(batch_reshape, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    yield from tfds.as_numpy(ds)