from rai_wm import embed, unembed
import time
import string
import random
import cv2

def generate_hex_string(length=38):
    hex_chars = string.hexdigits  # Строка, содержащая все шестнадцатеричные символы
    return ''.join(random.choice(hex_chars).lower() for _ in range(length))

arrr = [
    ('./1.jpg', './1w.jpg', f'abc{str("ab41c9e799ba772a4a4c04b4efa88fb2")}123'),
]

CROP_COEF = 64

WAVELETS = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'haar', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']

for wavelet in WAVELETS:
    print(wavelet)
    for cover_img_path, watermarked_img_path, watermark_str in arrr:
        start_time = time.time()

        embed(cover_img_path, watermarked_img_path, watermark_str, 25645, wavelet)
        wm_text = unembed(watermarked_img_path, 25645, pattern=(watermark_str[:3], watermark_str[-3:]), wavelet=wavelet)
        # print(f'{watermark_str}\n{wm_text}')
        print("!MATCH1") if watermark_str == wm_text else None

        image_full = cv2.imread(watermarked_img_path, flags=cv2.IMREAD_UNCHANGED)
        height, width = image_full.shape[:2]
        image_cropped = image_full[CROP_COEF:height, 0:width]
        cv2.imwrite(f"{watermarked_img_path}-edited.png", image_cropped)
        wm_text = unembed(f"{watermarked_img_path}-edited.png", 25645, pattern=(watermark_str[:3], watermark_str[-3:]), wavelet=wavelet)
        # print(f'{watermark_str}\n{wm_text}')
        print("!MATCH2") if watermark_str == wm_text else None

        end_time = time.time()  # Record time after the block ends
        execution_time = end_time - start_time