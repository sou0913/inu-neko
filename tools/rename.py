# 画像ファイル名変換

import glob
files = glob.glob('kamo_materials/**/*')
import os
import pathlib
for i, f in enumerate(files):
    path = pathlib.Path(f)
    os.rename(f, str(path.parent) + "/" + str(i) + ".jpg")

