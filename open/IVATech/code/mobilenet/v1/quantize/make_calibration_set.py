import click
import os.path
from PIL import Image
import numpy as np
from iva_applications.inception_v1.preprocess import image_to_tensor

home = os.path.expanduser("~")

@click.command()
@click.option('-d', '--dataset', default=os.path.join(home, "datasets", "imagenet"), help='Path to ImageNet dataset', type=click.Path(exists=True, file_okay=False))
@click.option('-c', '--calibration-list', required=True, help="ImageNet calibration file list", type=click.File())
@click.option('-o', '--output', default='calibration_tensors.npy', help="Calibration tensors file for tpu_quantize program", type=click.File(mode="wb"))
def make_calibration_set(dataset, calibration_list, output):
    tensors = []
    for image_name in calibration_list.read().splitlines():
        path = os.path.join(dataset, image_name)
        image = Image.open(path)
        tensor = image_to_tensor(image)
        tensors.append(tensor)
    tensors = np.array(tensors, dtype='float32')
    np.save(output, tensors)


if __name__ == "__main__":
    make_calibration_set()