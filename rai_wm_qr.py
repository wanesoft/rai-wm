import cv2
import copy
import numpy as np
from numpy.linalg import svd
from cv2 import dct, idct
from pywt import dwt2, idwt2
from collections import Counter
import math
from pyzbar.pyzbar import decode

from aztec_code_generator import AztecCode
import zxing

alpha1, alpha2 = 58, 30
block_size = np.array([4, 4])

def _split_image_from_center(image):
    height, width = image.shape[:2]
    
    center_y, center_x = height // 2, width // 2
    
    start_y = center_y - (center_y % 128)
    start_x = center_x - (center_x % 128)
    
    block_size = 128
    sub_images = []
    
    for i in range(start_y, -1, -block_size):
        for j in range(start_x, -1, -block_size):
            if i + block_size <= height and j + block_size <= width:
                sub_image = image[i:i+block_size, j:j+block_size]
                sub_images.append((i, j, sub_image))
    
    for i in range(start_y, height, block_size):
        for j in range(start_x, -1, -block_size):
            if i + block_size <= height and j + block_size <= width:
                sub_image = image[i:i+block_size, j:j+block_size]
                sub_images.append((i, j, sub_image))
    
    for i in range(start_y, -1, -block_size):
        for j in range(start_x, width, block_size):
            if i + block_size <= height and j + block_size <= width:
                sub_image = image[i:i+block_size, j:j+block_size]
                sub_images.append((i, j, sub_image))
    
    for i in range(start_y, height, block_size):
        for j in range(start_x, width, block_size):
            if i + block_size <= height and j + block_size <= width:
                sub_image = image[i:i+block_size, j:j+block_size]
                sub_images.append((i, j, sub_image))

    return sub_images

def _reassemble_image(sub_images, original_shape, output_path):
    reassembled_image = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)
    
    for (i, j, sub_image) in sub_images:
        reassembled_image[i:i+128, j:j+128] = sub_image
                
    return reassembled_image

def _hex_to_bits(hex_string):
    scale = 16
    num_of_bits = 4
    bits_string = ""
    for char in hex_string:
        bits_string += bin(int(char, scale))[2:].zfill(num_of_bits)
    return bits_string

def _bits_to_hex(bits_string):
    hex_string = ""
    for i in range(0, len(bits_string), 4):
        bits_chunk = bits_string[i:i+4]
        hex_string += hex(int(bits_chunk, 2))[2:]
    return hex_string

def embed(cover_path, watermarked_path, wm, pwd: int):
    image_full = cv2.imread(cover_path, flags=cv2.IMREAD_UNCHANGED)
    
    # sub_images = _split_image_from_center(image_full)
    sub_images = [image_full]

    sub_images_covered = []
    for sub_image in sub_images:
        image_raw = sub_image.astype(np.float32)
        image_size = image_raw.shape[:2]
        wm_size, block_num = 0, 0
        coeffi_approx = [np.array([])] * 3
        hvds = [np.array([])] * 3
        coeffi_approx_block = [np.array([])] * 3
        coeffi_approx_part = [np.array([])] * 3

        # byte = _hex_to_bits(wm)
        # wm_bits = (np.array(list(byte)) == '1')
        # wm_size = wm_bits.size

        data = wm
        print(f'income: {data}')

        aztec_code = AztecCode(data)

        img = aztec_code.image(module_size=4, border=1)
        img = img.resize((32, 32))
        img.save("aztec_code.png")

        reader = zxing.BarCodeReader()
        barcode = reader.decode("aztec_code.png")
        print(f'before: {barcode.parsed}')

        wm = cv2.imread(filename="aztec_code.png", flags=cv2.IMREAD_GRAYSCALE)
        assert wm is not None
        wm_bits = wm.flatten() > 128
        wm_size = wm_bits.size

        # np.random.RandomState(pwd).shuffle(wm_bits)

        coeffi_approx_size = [(i + 1) // 2 for i in image_size]

        coeffi_approx_block_size = (coeffi_approx_size[0] // block_size[0], coeffi_approx_size[1] // block_size[1],
                                    block_size[0], block_size[1])
        strides = 4 * np.array([coeffi_approx_size[1] * block_size[0], block_size[1], coeffi_approx_size[1], 1])

        image_YUV = cv2.copyMakeBorder(cv2.cvtColor(image_raw, cv2.COLOR_BGR2YUV),
                                    0, image_raw.shape[0] % 2, 0, image_raw.shape[1] % 2,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))

        for channel in range(3):
            coeffi_approx[channel], hvds[channel] = dwt2(image_YUV[:, :, channel], 'haar')
            coeffi_approx_block[channel] = np.lib.stride_tricks.as_strided(
                coeffi_approx[channel].astype(np.float32), coeffi_approx_block_size, strides
            )

        block_num = coeffi_approx_block_size[0] * coeffi_approx_block_size[1]

        print(f'{wm_size}<{block_num}')

        assert wm_size < block_num, IndexError(
            'the watermark is to big for the picture: {}kb < {}kb'.format(block_num / 1000, wm_size / 1000))

        part_size = coeffi_approx_block_size[:2] * block_size

        blocks_index = [(i, j) for i in range(coeffi_approx_block_size[0]) for j in range(coeffi_approx_block_size[1])]

        shuffler_arr = np.random.RandomState(pwd).random(size=(block_num, block_size[0] * block_size[1])).argsort(axis=1)

        covered_coeffi_approx = copy.deepcopy(coeffi_approx)
        covered_YUV = [np.array([])] * 3

        for channel in range(3):
            tmp_res = []
            for i in range(block_num):
                block, shuffler = coeffi_approx_block[channel][blocks_index[i]], shuffler_arr[i]

                wm_bit = wm_bits[i % wm_size]

                block_dct = dct(block)
                # block_dct_shuffled = block_dct.flatten()[shuffler].reshape(block_size)
                block_dct_shuffled = block_dct

                u, s, v = svd(block_dct_shuffled)

                s[0] = (s[0] // alpha1 + 1 / 4 + 1 / 2 * wm_bit) * alpha1
                s[1] = (s[1] // alpha2 + 1 / 4 + 1 / 2 * wm_bit) * alpha2

                block_dct_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
                # block_dct_flatten[shuffler] = block_dct_flatten.copy()

                var = idct(block_dct_flatten.reshape(block_size))

                tmp_res.append(var)

            for i in range(block_num):
                coeffi_approx_block[channel][blocks_index[i]] = tmp_res[i]

            coeffi_approx_part[channel] = np.concatenate(np.concatenate(coeffi_approx_block[channel], 1), 1)
            covered_coeffi_approx[channel][:part_size[0], :part_size[1]] = coeffi_approx_part[channel]
            covered_YUV[channel] = idwt2((covered_coeffi_approx[channel], hvds[channel]), "haar")

        covered_image_YUV = np.stack(covered_YUV, axis=2)
        covered_image_YUV = covered_image_YUV[:image_size[0], :image_size[1]]

        covered_img = cv2.cvtColor(covered_image_YUV, cv2.COLOR_YUV2BGR)
        covered_img = np.clip(covered_img, a_min=0, a_max=255)

        # sub_images_covered.append((sub_image[0], sub_image[1], covered_img))
        sub_images_covered.append(covered_img)

    # ret = _reassemble_image(sub_images_covered, image_full.shape, watermarked_path)
    # cv2.imwrite(watermarked_path, ret)
    cv2.imwrite(watermarked_path, sub_images_covered[0])

def adjust_aspect_ratio(image_path, known_ratios=[(2, 3), (4, 3), (16, 9), (21, 9)]):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    gcd = math.gcd(width, height)
    current_ratio = (width // gcd, height // gcd)

    best_ratio = min(known_ratios, key=lambda ratio: abs(ratio[0] / ratio[1] - current_ratio[0] / current_ratio[1]))

    if current_ratio != best_ratio:
        new_width = width if best_ratio[0] / best_ratio[1] > current_ratio[0] / current_ratio[1] else int(height * best_ratio[0] / best_ratio[1])
        new_height = height if best_ratio[0] / best_ratio[1] < current_ratio[0] / current_ratio[1] else int(width * best_ratio[1] / best_ratio[0])
        image = cv2.resize(image, (new_width, new_height))

    return image

def unembed(wm_pic_path, pwd: int, pattern=('666', '666')):
    image_full = cv2.imread(wm_pic_path, flags=cv2.IMREAD_UNCHANGED)
    
    # sub_images = _split_image_from_center(image_full)
    sub_images = [image_full]

    wms_from_blocks = []
    wms_from_blocks_alter = []
    for sub_image in sub_images:
        wm_size = 152
        wm_size2 = (32, 32)
        wm_size = np.array(wm_size2).prod()       
        
        block_num = 0
        coeffi_approx = [np.array([])] * 3
        hvds = [np.array([])] * 3
        coeffi_approx_block = [np.array([])] * 3

        image_raw = sub_image.astype(np.float32)
        image_size = image_raw.shape[:2]

        coeffi_approx_size = [(i + 1) // 2 for i in image_size]

        coeffi_approx_block_size = (coeffi_approx_size[0] // block_size[0], coeffi_approx_size[1] // block_size[1],
                                    block_size[0], block_size[1])
        strides = 4 * np.array([coeffi_approx_size[1] * block_size[0], block_size[1], coeffi_approx_size[1], 1])

        image_YUV = cv2.copyMakeBorder(cv2.cvtColor(image_raw, cv2.COLOR_BGR2YUV),
                                       0, image_raw.shape[0] % 2, 0, image_raw.shape[1] % 2,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))

        for channel in range(3):
            coeffi_approx[channel], hvds[channel] = dwt2(image_YUV[:, :, channel], 'haar')
            coeffi_approx_block[channel] = np.lib.stride_tricks.as_strided(
                coeffi_approx[channel].astype(np.float32), coeffi_approx_block_size, strides
            )

        block_num = coeffi_approx_block_size[0] * coeffi_approx_block_size[1]
        # assert wm_size < block_num, IndexError(
            # 'the watermark is to big for the picture: {}kb < {}kb'.format(block_num / 1000, wm_size / 1000))

        block_index = [(i, j) for i in range(coeffi_approx_block_size[0]) for j in range(coeffi_approx_block_size[1])]

        wm_size = np.array(wm_size).prod()

        wm_blocks_bit = np.zeros(shape=(3, block_num))

        shuffler_arr = np.random.RandomState(pwd).random(size=(block_num, block_size[0] * block_size[1])).argsort(axis=1)

        wm_all_bits = []
        for channel in range(3):
            tmp_res = []
            for i in range(block_num):
                block, shuffler, i = coeffi_approx_block[channel][block_index[i]], shuffler_arr[i], i

                # block_dct_shuffled = dct(block).flatten()[shuffler].reshape(block_size)
                block_dct_shuffled = dct(block)

                _, s, _ = svd(block_dct_shuffled)

                wm = (s[0] % alpha1 > alpha1 / 2) * 1
                tmp = (s[1] % alpha2 > alpha2 / 2) * 1
                wm = (wm * 3 + tmp * 1) / 4
                tmp_res.append(wm)
                wm_all_bits.append(wm)

            wm_blocks_bit[channel, :] = tmp_res

        wm_draft = np.zeros(shape=wm_size)
        for i in range(wm_size):
            wm_draft[i] = wm_blocks_bit[:, i::wm_size].mean()

        wm = 255 * wm_draft.reshape(wm_size2[0], wm_size2[1])
        cv2.imwrite('wmm2.png', wm)

        reader = zxing.BarCodeReader()
        barcode = reader.decode("wmm2.png")

        return barcode.parsed

        # wm_index = np.arange(wm_size)
        # np.random.RandomState(pwd).shuffle(wm_index)
        # wm_draft[wm_index] = wm_draft.copy()

        byte = ''.join(str((i >= 0.5) * 1) for i in wm_draft)
        wm = _bits_to_hex(byte)
        wms_from_blocks.append(wm)

        byte = ''.join(str((i >= 0.5) * 1) for i in wm_all_bits)
        wm = _bits_to_hex(byte)
        wms_from_blocks_alter.append(wm)

    string_counts = Counter(wms_from_blocks)
    most_common, _ = string_counts.most_common(1)[0]
    if not most_common.startswith(pattern[0]):
        wms_from_blocks.clear()
        for cur in wms_from_blocks_alter[0].split(pattern[0]):
            if len(cur) == 32:
                wms_from_blocks.append(cur)

        string_counts = Counter(wms_from_blocks)
        most_common, _ = string_counts.most_common(1)[0] if len(wms_from_blocks) else 'none', None
        most_common = pattern[0] + most_common[0] + pattern[1]

    return most_common