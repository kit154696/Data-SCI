"""
Setup Thai Font for Matplotlib - Final Fix
===========================================
แก้ปัญหาภาษาไทยแบบสมบูรณ์โดยโหลดฟอนต์โดยตรง
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

print("=" * 60)
print("🔧 กำลังแก้ปัญหาฟอนต์ภาษาไทยแบบถาวร")
print("=" * 60)

# โหลดฟอนต์ Thonburi โดยตรง
font_path = '/System/Library/Fonts/Supplemental/Thonburi.ttc'
if Path(font_path).exists():
    # เพิ่มฟอนต์เข้า matplotlib
    fm.fontManager.addfont(font_path)
    print(f"✓ โหลดฟอนต์: {font_path}")
else:
    print(f"✗ ไม่พบฟอนต์: {font_path}")

# ตั้งค่า matplotlib ให้ใช้ Thonburi
plt.rcParams['font.family'] = 'Thonburi'
plt.rcParams['axes.unicode_minus'] = False

print("\n📊 ทดสอบการแสดงภาษาไทย...")

# สร้างกราฟทดสอบ
fig, ax = plt.subplots(figsize=(10, 6))

# ทดสอบข้อความภาษาไทย
test_texts = [
    'ความสัมพันธ์ระหว่างตัวแปรสำคัญ',
    'การกระจายของคะแนน',
    'จำนวนนักศึกษา',
    'คะแนนเฉลี่ย',
    'ผลการวิเคราะห์'
]

for i, text in enumerate(test_texts):
    ax.text(0.5, 0.8 - i*0.15, text,
           fontsize=14, ha='center', va='center',
           transform=ax.transAxes)

ax.set_title('ทดสอบการแสดงภาษาไทยใน Matplotlib', fontsize=16, pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig('analysis_output/font_test_thonburi.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ บันทึก: analysis_output/font_test_thonburi.png")

# สร้างไฟล์ตั้งค่าสำหรับโปรเจค
config_content = """# Matplotlib Thai Font Configuration
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
"""

with open('thai_font_config.py', 'w', encoding='utf-8') as f:
    f.write(config_content)

print("\n✅ สร้างไฟล์ตั้งค่า: thai_font_config.py")

print("\n" + "=" * 60)
print("✅ แก้ไขเสร็จสิ้น!")
print("=" * 60)
print("\n📋 วิธีใช้:")
print("  1. เปิดดู font_test_thonburi.png เพื่อดูว่าภาษาไทยแสดงได้แล้ว")
print("  2. ใส่โค้ดนี้ที่ด้านบนของทุกสคริปต์:")
print()
print("     from thai_font_config import *")
print()
print("  3. หรือ copy โค้ดจาก thai_font_config.py ไปวางในสคริปต์")
print("\n" + "=" * 60)
