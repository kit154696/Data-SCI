# Matplotlib Thai Font Configuration
# วางโค้ดนี้ไว้ด้านบนของสคริปต์ก่อน import pyplot

import matplotlib.font_manager as fm
from pathlib import Path

# โหลดฟอนต์ Thonburi
font_path = '/System/Library/Fonts/Supplemental/Thonburi.ttc'
if Path(font_path).exists():
    fm.fontManager.addfont(font_path)

import matplotlib.pyplot as plt

# ตั้งค่าฟอนต์
plt.rcParams['font.family'] = 'Thonburi'
plt.rcParams['axes.unicode_minus'] = False
