# ui/utils/helpers.py
import cv2
import numpy as np
from PIL import Image, ImageTk


def parse_matrix(matrix_str, rows, cols):
    try:
        row_strs = matrix_str.split(";")
        matrix = []
        for r in row_strs:
            values = [float(x.strip()) for x in r.split(",")]
            matrix.append(values)
        arr = np.array(matrix, dtype=np.float32)
        if arr.shape != (rows, cols):
            raise ValueError(f"Matrix shape mismatch; expected ({rows}, {cols})")
        return arr
    except Exception as e:
        print("Error parsing matrix:", e)
        return None


def cv2_to_tk(cv_img):
    # Convert BGR to RGB and then to Tkinter PhotoImage.
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    return ImageTk.PhotoImage(im)
