"""Tool for creating multi-resolution TFRecords datasets for StyleGAN and ProGAN."""

import argparse
import glob
import sys
import threading
import traceback

import PIL.Image
import numpy as np
# pylint: disable=too-many-lines
import os
import six.moves.queue as Queue  # pylint: disable=import-error
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import dnnlib.tflib as tflib
from training import dataset
#from scipy.misc import imresize


# ----------------------------------------------------------------------------


def error(msg):
    print("Error: " + msg)
    exit(1)


# ----------------------------------------------------------------------------


class TFRecordExporter:
    def __init__(
            self, tfrecord_dir, tfr_prefix, expected_images, height, width, print_progress=True, progress_interval=10
    ):
        self.tfrecord_dir = tfrecord_dir
        self.tfr_prefix = tfr_prefix
        self.expected_images = expected_images
        self.cur_images = 0
        self.shape = None
        self.height = height
        self.width = width
        self.tfr_writers = []
        self.print_progress = print_progress
        self.progress_interval = progress_interval

        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print("%-40s\r" % "Flushing data...", end="", flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print("%-40s\r" % "", end="", flush=True)
            print("Added %d images." % self.cur_images)

    def choose_shuffled_order(
            self, seed=123
    ):  # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(seed).shuffle(order)
        return order

    def store_image(self, encoded_jpg):

        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print(
                "%d / %d\r" % (self.cur_images, self.expected_images),
                end="",
                flush=True,
            )
        for lod, tfr_writer in enumerate(self.tfr_writers):
            ex = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "shape" : tf.train.Feature(
                            int64_list=tf.train.Int64List(value=self.shape)
                        ),
                        "data" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg]))
                    }
                )
            )
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1
        
    def create_tfr_writer(self, shape):
        self.shape = [shape[2], shape[0], shape[1]]
        assert self.shape[0] in [1, 3]
        assert self.shape[1] == self.height
        assert self.shape[2] == self.width
        # assert self.shape[2] == self.width
        tfr_opt = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.NONE
        )
        resolution_log2 = int(np.log2(self.height))
        tfr_file = self.tfr_prefix + "-r%02d.tfrecords" % (
                    resolution_log2
        )
        self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))




    def add_labels(self, labels):
        if self.print_progress:
            print("%-40s\r" % "Saving labels...", end="", flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + "-rxx.labels", "wb") as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ----------------------------------------------------------------------------


class ExceptionInfo(object):
    def __init__(self):
        self.value = sys.exc_info()[1]
        self.traceback = traceback.format_exc()


# ----------------------------------------------------------------------------


class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))


# ----------------------------------------------------------------------------


class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = Queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for _idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        assert hasattr(func, "__call__")  # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = Queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func):  # returns (result, args)
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            print("\n\nWorker thread caught an exception:\n" + result.traceback)
            raise result.value
        return result, args

    def finish(self):
        for _idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self):  # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.finish()

    def process_items_concurrently(
            self,
            item_iterator,
            process_func=lambda x: x,
            pre_func=lambda x: x,
            post_func=lambda x: x,
            max_items_in_flight=None,
    ):
        if max_items_in_flight is None:
            max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(in_prepared, _idx):
            return process_func(in_prepared)

        def retire_result():
            processed, (_prepared, in_idx) = self.get_result(task_func)
            results[in_idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1

        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result():
                    yield res
        while retire_idx[0] < len(results):
            for res in retire_result():
                yield res


# ----------------------------------------------------------------------------


def display(tfrecord_dir):
    print('Loading dataset "%s"' % (tfrecord_dir))
    tflib.init_tf({"gpu_options.allow_growth": True})
    dset = dataset.TFRecordDataset(
        tfrecord_dir, max_label_size="full", repeat=False, shuffle_mb=0
    )
    tflib.init_uninitialized_vars()
    import cv2  # pip install opencv-python

    idx = 0
    while True:
        try:
            images, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if idx == 0:
            print("Displaying images", flush=True)
            cv2.namedWindow("dataset_tool")
            print("Press SPACE or ENTER to advance, ESC to exit")
        print("\nidx = %-8d\nlabel = %s" % (idx, labels[0].tolist()), flush=True)
        cv2.imshow(
            "dataset_tool", images[0].transpose(1, 2, 0)[:, :, ::-1]
        )  # CHW => HWC, RGB => BGR
        idx += 1
        if cv2.waitKey() == 27:
            break
    print("\nDisplayed %d images." % idx, flush=True)


# ----------------------------------------------------------------------------


def extract(tfrecord_dir, output_dir):
    print('Loading dataset "%s"' % tfrecord_dir)
    tflib.init_tf({"gpu_options.allow_growth": True})
    dset = dataset.TFRecordDataset(
        tfrecord_dir, max_label_size=0, repeat=False, shuffle_mb=0
    )
    tflib.init_uninitialized_vars()

    print('Extracting images to "%s"' % output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    idx = 0
    while True:
        if idx % 10 == 0:
            print("%d\r" % idx, end="", flush=True)
        try:
            images, _labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if images.shape[1] == 1:
            img = PIL.Image.fromarray(images[0][0], "L")
        else:
            img = PIL.Image.fromarray(images[0].transpose(1, 2, 0), "RGB")
        img.save(os.path.join(output_dir, "img%08d.png" % idx))
        idx += 1
    print("Extracted %d images." % idx)


# ----------------------------------------------------------------------------


def compare(tfrecord_dir_a, tfrecord_dir_b, ignore_labels):
    max_label_size = 0 if ignore_labels else "full"
    print('Loading dataset "%s"' % tfrecord_dir_a)
    tflib.init_tf({"gpu_options.allow_growth": True})
    dset_a = dataset.TFRecordDataset(
        tfrecord_dir_a, max_label_size=max_label_size, repeat=False, shuffle_mb=0
    )
    print('Loading dataset "%s"' % tfrecord_dir_b)
    dset_b = dataset.TFRecordDataset(
        tfrecord_dir_b, max_label_size=max_label_size, repeat=False, shuffle_mb=0
    )
    tflib.init_uninitialized_vars()

    print("Comparing datasets")
    idx = 0
    identical_images = 0
    identical_labels = 0
    while True:
        if idx % 100 == 0:
            print("%d\r" % idx, end="", flush=True)
        try:
            images_a, labels_a = dset_a.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_a, labels_a = None, None
        try:
            images_b, labels_b = dset_b.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_b, labels_b = None, None
        if images_a is None or images_b is None:
            if images_a is not None or images_b is not None:
                print("Datasets contain different number of images")
            break
        if images_a.shape == images_b.shape and np.all(images_a == images_b):
            identical_images += 1
        else:
            print("Image %d is different" % idx)
        if labels_a.shape == labels_b.shape and np.all(labels_a == labels_b):
            identical_labels += 1
        else:
            print("Label %d is different" % idx)
        idx += 1
    print("Identical images: %d / %d" % (identical_images, idx))
    if not ignore_labels:
        print("Identical labels: %d / %d" % (identical_labels, idx))


def _get_all_files(path):
    if os.path.isfile(path):
        return [path]

    possible_files = sorted(glob.glob(os.path.join(path, "*")))
    return_list = []
    for possible_file in possible_files:
        return_list.extend(_get_all_files(possible_file))
    return return_list


# ----------------------------------------------------------------------------

def create_aydao(output_tfrecord_dir, input_image_dir, dataset_name, height, width, shuffle=True):

    out_path = os.path.join(output_tfrecord_dir, dataset_name)
    os.makedirs(out_path, exist_ok=True)
    tfr_prefix = os.path.join(out_path, dataset_name)

    print('Loading images from "%s"' % input_image_dir)
    image_filenames = _get_all_files(input_image_dir)
    print(f"detected {len(image_filenames)} images ...")
    if len(image_filenames) == 0:
        error("No input images found")
    
    first_filename = image_filenames[0]
    first = PIL.Image.open(first_filename)
    image = np.asarray(first)
    assert image.ndim == 3
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    assert height == width # enforced for now
    assert height in [2**x for x in range(2,11)] # max is 1024

    if channels not in [1, 3]:
        error("Input images must be stored as RGB or grayscale")
    seed = 123
    if shuffle:
        seed = seed if shuffle == 1 else shuffle # allow user to specify seed value in shuffle (note: can't use a seed of '1')
        print("Shuffle the images... using seed", seed)
    with TFRecordExporter(output_tfrecord_dir, tfr_prefix, len(image_filenames), height, width) as tfr:
        order = (
            tfr.choose_shuffled_order(seed) if shuffle else np.arange(len(image_filenames))
        )
        tfr.create_tfr_writer(image.shape)
        print("Adding the images to tfrecords ...")
        for idx in range(order.size):
            if idx % 1000 == 0:
                print ("added images", idx, flush=True)
            with tf.gfile.FastGFile(image_filenames[order[idx]], 'rb') as fid:
                try:
                    tfr.store_image(fid.read())
                except:
                    print ('error when adding', image_filenames[order[idx]])
                    continue

# ----------------------------------------------------------------------------


def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Tool for creating multi-resolution TFRecords datasets for StyleGAN and ProGAN.",
        epilog='Type "%s <command> -h" for more information.' % prog,
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    def add_command(cmd, desc, example=None):
        epilog = "Example: %s %s" % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command("display", "Display images in dataset.", "display datasets/mnist")
    p.add_argument("tfrecord_dir", help="Directory containing dataset")

    p = add_command(
        "extract", "Extract images from dataset.", "extract datasets/mnist mnist-images"
    )
    p.add_argument("tfrecord_dir", help="Directory containing dataset")
    p.add_argument("output_dir", help="Directory to extract the images into")

    p = add_command(
        "compare", "Compare two datasets.", "compare datasets/mydataset datasets/mnist"
    )
    p.add_argument("tfrecord_dir_a", help="Directory containing first dataset")
    p.add_argument("tfrecord_dir_b", help="Directory containing second dataset")
    p.add_argument(
        "--ignore_labels", help="Ignore labels (default: 0)", type=int, default=0
    )

    p = add_command(
        "create_aydao",
        "Create dataset from a directory full of images. Please be careful"
        "since the tool recursively searches inside every sub-directory for image files",
        "create_from_images_raw datasets/mydataset myimagedir",
    )
    p.add_argument("output_tfrecord_dir", help="New dataset directory to be created")
    p.add_argument("input_image_dir", help="Directory containing the images")
    p.add_argument("dataset_name", help="Prefix for tfrecords file")
    p.add_argument(
        "height",
        help="image width and height should be power of 2",
        type=int
    )
    p.add_argument(
        "width",
        help="image width and height should be power of 2",
        type=int
    )
    p.add_argument(
        "--shuffle", help="Randomize image order (default: 1)", type=int, default=1
    )

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ["-h"])
    func = globals()[args.command]
    del args.command
    func(**vars(args))


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)

# ----------------------------------------------------------------------------
