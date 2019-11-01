import cv2
import numpy as np
import re
import io
from PIL import Image, ImageCms



FLOW_CHANNELS = 2
FLOW_TAG_FLOAT = 202021.25


def load_pfm(file_path):
    """
    Load a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    :param file_path: Str.
    :return: Np array of shape (H, W, C).
    """
    with open(file_path, 'rb') as file:
        header = file.readline().decode('latin-1').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().decode('latin-1').rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width, 1)
        return np.reshape(data, shape), scale


def read_flow_file(file_name, extras=None, use_default_valid=False):
    """
    Supports .pfm, .flo, or .png extensions.
    The .png extension assumes the Kitti format.
    :param file_name: Str.
    :param extras: Optional dict.
        A valid mask is returned under the 'valid' key if there is one (i.e. if the format is png). Otherwise, the
        'valid' key will return None.
    :param use_default_valid: Bool. If True, extras['valid'] will return ones (true) by default if the file was valid.
        Otherwise, None will be returned if no valid mask was found in the flow file.
    :return: Numpy array of shape (height, width, FLOW_CHANNELS).
        Returns None if the tag in the file header was invalid.
    """
    if extras is None:
        extras = {}
    extras['valid'] = None
    if file_name.endswith('.flo'):
        with open(file_name, 'rb') as file:
            tag = np.fromfile(file, dtype=np.float32, count=1)[0]
            if tag != FLOW_TAG_FLOAT:
                return None
            width = np.fromfile(file, dtype=np.int32, count=1)[0]
            height = np.fromfile(file, dtype=np.int32, count=1)[0]

            num_image_floats = width * height * FLOW_CHANNELS
            image = np.fromfile(file, dtype=np.float32, count=num_image_floats)
            image = image.reshape((height, width, FLOW_CHANNELS))
            final_flow = image
    elif file_name.endswith('.pfm'):
        np_array, scale = load_pfm(file_name)
        if scale != 1.0:
            raise Exception('Pfm flow scale must be 1.0.')
        # PFM file convention has the y axis going upward. This is unconventional, so we need to flip it.
        # Also, PFM can only have 1 or 3 colour channels. For flow, the last channel is set to 0.0.
        final_flow = np.flip(np_array[..., 0:2], axis=0)
    elif file_name.endswith('.png'):
        raw = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if raw is None:
            return None
        assert raw.dtype == np.uint16
        flow = (raw[..., :2].astype(np.float32) - (2 ** 15)) / 64.0
        valid = raw[..., 2:].astype(np.bool)
        assert valid.shape[2] == 1
        extras['valid'] = valid
        final_flow = flow * valid.astype(np.float32)
    else:
        raise Exception('Not a supported flow format.')
    if extras['valid'] is None and use_default_valid:
        height, width = final_flow.shape[0], final_flow.shape[1]
        extras['valid'] = np.ones((height, width, 1), dtype=np.bool)
    return final_flow


def pad_image_for_divisibility(image, n):
    """
    Pads an image with zeros such that the dimensions can be divisible by 2 at least n times.
    If the dimensions can already be sufficiently divisible by 2, it does nothing.
    :param image: Numpy array of shape [H, W, C].
    :param n: Int. Number of times the resolution needs to be divisible by 2.
    :return: Numpy array of shape [H_d, W_d, C] where H_d, W_d can be divisible by 2 n times.
    """
    original_height, original_width = image.shape[0], image.shape[1]
    height = int((2 ** n) * np.ceil(original_height / (2 ** n)))
    width = int((2 ** n) * np.ceil(original_width / (2 ** n)))
    delta_h = height - original_height
    delta_w = width - original_width
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # Maintain the columns.
    padded = np.reshape(padded, (padded.shape[0], padded.shape[1], image.shape[2]))
    return padded


def read_image(img_path, as_float=False):
    """
    :param img_path: Str.
    :param as_float: Bool. If true, then return the image as floats between [0, 1] instead of uint8s between [0, 255].
    :return: Numpy array of shape [H, W, 3].
    """
    # This is to handle image ICC profiles. To simplify things we convert all images to standard RGB (sRGB).
    # https://stackoverflow.com/questions/50622180/does-pil-image-convertrgb-converts-images-to-srgb-or-adobergb
    def _convert_to_srgb(img):
        '''Convert PIL image to sRGB color space (if possible)'''
        icc = img.info.get('icc_profile', '')
        if icc:
            io_handle = io.BytesIO(icc)  # virtual file
            src_profile = ImageCms.ImageCmsProfile(io_handle)
            dst_profile = ImageCms.createProfile('sRGB')
            img = ImageCms.profileToProfile(img, src_profile, dst_profile)
        return img

    img = Image.open(img_path)
    img = np.array(_convert_to_srgb(img))
    if len(img.shape) == 2:
        img = np.stack(3 * [img], axis=-1)

    # Discard the alpha channel.
    if img.shape[2] == 4:
        img = img[..., :3]

    if as_float:
        return img.astype(dtype=np.float32) / 255.0
    else:
        return img
