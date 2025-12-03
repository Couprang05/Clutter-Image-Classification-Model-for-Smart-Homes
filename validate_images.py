from pathlib import Path
from PIL import Image, ImageFile
import imghdr

p = Path(r"D:\University\SEM-V\Deep Learning\DL Project\Clutter_Image_Classification_Model_for_Smart_Homes\dataset\raw\indoorCVPR_09\Images\laundromat\Laundry_Room_bmp.jpg")
print("exists:", p.exists(), "size(bytes):", p.stat().st_size)
print("imghdr type:", imghdr.what(str(p)))
ImageFile.LOAD_TRUNCATED_IMAGES = False
try:
    with Image.open(p) as im:
        print("PIL opened. format, mode, size:", im.format, im.mode, im.size)
        im.verify()
        print("PIL verify: OK")
except Exception as e:
    print("PIL error:", e)
