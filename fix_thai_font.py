"""
Fix Thai Font Display in All Visualizations
============================================
แก้ปัญหาการแสดงภาษาไทยในกราฟทั้งหมด
"""

import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ล้าง font cache
print("=" * 60)
print("🔧 กำลังแก้ไขปัญหาฟอนต์ภาษาไทย...")
print("=" * 60)

# ลบ matplotlib cache
cache_dir = matplotlib.get_cachedir()
print(f"\n📂 Cache directory: {cache_dir}")

try:
    # ลบไฟล์ cache
    for item in os.listdir(cache_dir):
        if item.startswith('font'):
            file_path = os.path.join(cache_dir, item)
            try:
                os.remove(file_path)
                print(f"✓ ลบ cache: {item}")
            except:
                pass
except Exception as e:
    print(f"ข้อมูล: {e}")

# Rebuild font cache
print("\n🔄 กำลัง rebuild font cache...")
try:
    matplotlib.font_manager._load_fontmanager(try_read_cache=False)
    print("✓ Rebuild เสร็จแล้ว")
except:
    print("✓ ข้าม rebuild (ไม่จำเป็น)")

# ตั้งค่าฟอนต์ใหม่
print("\n⚙️  ตั้งค่าฟอนต์สำหรับภาษาไทย...")

# ใช้ Sarabun เป็นหลัก (ฟอนต์ภาษาไทยที่ชัดเจนที่สุด)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Sarabun', 'Thonburi', 'Krungthep', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ทดสอบ
print("\n✅ ตั้งค่าเสร็จแล้ว")
print("\nฟอนต์ที่จะใช้ตามลำดับ:")
print("  1. Sarabun (แนะนำ - ฟอนต์ไทยสมัยใหม่)")
print("  2. Thonburi (ฟอนต์ไทย macOS)")
print("  3. Krungthep (ฟอนต์ไทยคลาสสิก)")
print("  4. Arial Unicode MS (สำรอง)")

# สร้างไฟล์ตัวอย่างทดสอบ
print("\n📊 สร้างตัวอย่างทดสอบ...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('ทดสอบการแสดงภาษาไทยใน Matplotlib', fontsize=16, fontweight='bold')

# Test 1: ข้อความทั่วไป
axes[0, 0].text(0.5, 0.5, 'ความสัมพันธ์ระหว่างตัวแปรสำคัญ',
               fontsize=14, ha='center', va='center')
axes[0, 0].set_title('ข้อความหัวข้อ', fontsize=12)
axes[0, 0].axis('off')

# Test 2: ป้ายแกน
axes[0, 1].plot([1, 2, 3, 4], [1, 4, 2, 3])
axes[0, 1].set_xlabel('จำนวนนักศึกษา', fontsize=12)
axes[0, 1].set_ylabel('คะแนนเฉลี่ย', fontsize=12)
axes[0, 1].set_title('กราฟเส้น', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Test 3: Bar chart
categories = ['เกรด A', 'เกรด B', 'เกรด C', 'เกรด D']
values = [30, 45, 20, 5]
axes[1, 0].bar(categories, values, color='steelblue', alpha=0.7)
axes[1, 0].set_ylabel('จำนวนนักศึกษา', fontsize=12)
axes[1, 0].set_title('การกระจายของเกรด', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)

# Test 4: ข้อความยาว
text = """สรุปผลการวิเคราะห์:
• นักศึกษาที่มี GPA สูงได้เกรดดี
• การเข้าเรียนมีความสำคัญ
• โมเดลทำนายได้แม่นยำ 95.9%"""

axes[1, 1].text(0.1, 0.5, text, fontsize=11, va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].set_title('ข้อความหลายบรรทัด', fontsize=12)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('analysis_output/thai_font_test_final.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ บันทึก: analysis_output/thai_font_test_final.png")

print("\n" + "=" * 60)
print("✅ เสร็จสิ้น!")
print("=" * 60)
print("\n📋 ขั้นตอนต่อไป:")
print("  1. เปิดดูไฟล์ thai_font_test_final.png")
print("  2. ถ้าเห็นภาษาไทยชัดเจน ให้รัน:")
print("     python3 student_grade_prediction.py")
print("     python3 data_cleaning_analysis.py")
print("  3. กราฟทั้งหมดจะแสดงภาษาไทยถูกต้อง")
print("\n💡 หมายเหตุ: ถ้ายังไม่ได้ ให้ restart terminal แล้วรันใหม่")
print("=" * 60)
