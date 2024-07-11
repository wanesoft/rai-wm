from rai_wm import embed, unembed
import time
import string
import random
import cv2

def generate_hex_string(length=32):
    hex_chars = string.hexdigits  # Строка, содержащая все шестнадцатеричные символы
    return ''.join(random.choice(hex_chars).lower() for _ in range(length))

arrr = [
    # ('./small.png', './small2.png', '1'),
    # ('./2222.png', './2222w.png', '6655'),
    ('./x22.png', './x22w.png', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str("3ef4e2c64cd507a8e3649e129b024951")}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str(generate_hex_string())}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str("ec5ab6c13d4bd8959d2f07b1c46fea8b")}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str("3fcedcc3393b477fed70b09c4ab39337")}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str("9751539ad00843a6a0a5940054bf6d59")}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str("22dd6f90ea34ccd62b1b121beecd9e85")}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str("7c681667759b599407c43f240cdd3b16")}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str("d52de720708b05008459c438797c111a")}666'),
    # ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f"6b41c9e799ba772a4a4c04b4efa88fb2"),
    ('./IMG_1861.jpeg', './IMG_1861w.jpeg', f'666{str("ab41c9e799ba772a4a4c04b4efa88fb2")}666'),
]

for cover_img_path, watermarked_img_path, watermark_str in arrr:
    start_time = time.time()

    embed(cover_img_path, watermarked_img_path, watermark_str, 25645)
    wm_text = unembed(watermarked_img_path, 25645)
    print(f'{watermark_str}\n{wm_text}')
    print("!WARNING") if watermark_str != wm_text else None

    image_full = cv2.imread(watermarked_img_path, flags=cv2.IMREAD_UNCHANGED)
    height, width = image_full.shape[:2] 
    image_cropped = cv2.resize(image_full, (width, height-128))
    cv2.imwrite(f"{watermarked_img_path}-edited.png", image_cropped)
    wm_text = unembed(f"{watermarked_img_path}-edited.png", 25645)
    print(f'{watermark_str}\n{wm_text}')
    print("!WARNING") if watermark_str != wm_text else None

    image_full = cv2.imread(watermarked_img_path, flags=cv2.IMREAD_UNCHANGED)
    height, width = image_full.shape[:2] 
    image_cropped = cv2.resize(image_full, (int(width*1.01), int(height*1.0)))
    cv2.imwrite(f"{watermarked_img_path}-edited.png", image_cropped)
    wm_text = unembed(f"{watermarked_img_path}-edited.png", 25645)
    print(f'{watermark_str}\n{wm_text}')
    print("!WARNING") if watermark_str != wm_text else None

    image_full = cv2.imread(watermarked_img_path, flags=cv2.IMREAD_UNCHANGED)
    height, width = image_full.shape[:2] 
    image_cropped = image_full[32:height, 0:width]
    cv2.imwrite(f"{watermarked_img_path}-edited.png", image_cropped)
    wm_text = unembed(f"{watermarked_img_path}-edited.png", 25645)
    print(f'{watermark_str}\n{wm_text}')
    print("!WARNING") if watermark_str != wm_text else None

    image_full = cv2.imread(watermarked_img_path, flags=cv2.IMREAD_UNCHANGED)
    height, width = image_full.shape[:2] 
    image_cropped = image_full[32:height, 0:width]
    cv2.imwrite(f"{watermarked_img_path}-edited.png", image_cropped)
    wm_text = unembed(f"{watermarked_img_path}-edited.png", 25645)
    print(f'{watermark_str}\n{wm_text}')
    print("!WARNING") if watermark_str != wm_text else None

    end_time = time.time()  # Record time after the block ends
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")
    
    print()
