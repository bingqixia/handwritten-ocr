import gzip
import numpy


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(filename):
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
              'Invalid magic number %d in MNIST label file: %s' %
              (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return labels


def read_emnist(emnist_dir):
    print (emnist_dir)
    TRAIN_IMAGES = emnist_dir+'/emnist-byclass-train-images-idx3-ubyte.gz'
    TRAIN_LABELS = emnist_dir+'/emnist-byclass-train-labels-idx1-ubyte.gz'
    TEST_IMAGES = emnist_dir+'/emnist-byclass-test-images-idx3-ubyte.gz'
    TEST_LABELS = emnist_dir+'/emnist-byclass-test-labels-idx1-ubyte.gz'
    MAPPING = emnist_dir+'/emnist-byclass-mapping.txt'
    train_images = extract_images(TRAIN_IMAGES)
    train_labels = extract_labels(TRAIN_LABELS)
    test_images = extract_images(TEST_IMAGES)
    test_labels = extract_labels(TEST_LABELS)

    with open(MAPPING, "r") as f:
        mapping = f.readlines()
        mapping = {str(x.split()[0]): str(x.split()[1]) for x in mapping}

    # Convert to float32
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # Normalize
    train_images /= 255
    test_images /= 255

    # Output format: (28, 28, 1)
    return ((train_images, train_labels), (test_images, test_labels), mapping)
