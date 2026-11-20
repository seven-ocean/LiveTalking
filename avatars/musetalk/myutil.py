import numpy as np
import cv2
import copy
import os

# mask 羽化核大小（奇数，0/1=关闭）：边界硬切会把细微差异放大成边缘闪烁
_MASK_BLUR = int(os.getenv('MUSETALK_MASK_BLUR', '0') or 0)


def get_image_blending(image,face,face_box,mask_array,crop_box):
    body = image
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body[y_s:y_e, x_s:x_e].copy()
    face_large[y-y_s:y1-y_s, x-x_s:x1-x_s]=face

    if mask_array.ndim == 3:
        mask_image = cv2.cvtColor(mask_array,cv2.COLOR_BGR2GRAY)
    else:
        # 灰度掩码（read_imgs grayscale=True）直接使用，省 2/3 内存
        mask_image = mask_array
    mask_image = (mask_image/255).astype(np.float32)
    if _MASK_BLUR > 1:
        k = _MASK_BLUR if _MASK_BLUR % 2 == 1 else _MASK_BLUR + 1
        mask_image = cv2.GaussianBlur(mask_image, (k, k), 0)

    # mask_not = cv2.bitwise_not(mask_array)
    # prospect_tmp = cv2.bitwise_and(face_large, face_large, mask=mask_array)
    # background_img = body[y_s:y_e, x_s:x_e]
    # background_img = cv2.bitwise_and(background_img, background_img, mask=mask_not)
    # body[y_s:y_e, x_s:x_e] = prospect_tmp + background_img

    #print(mask_image.shape)
    #print(cv2.minMaxLoc(mask_image))

    body[y_s:y_e, x_s:x_e] = cv2.blendLinear(face_large,body[y_s:y_e, x_s:x_e],mask_image,1-mask_image)

    #body.paste(face_large, crop_box[:2], mask_image)
    return body