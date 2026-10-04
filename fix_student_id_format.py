"""
Fix StudentID Format Issue
===========================
แก้ปัญหา StudentID ถูกแปลงเป็นตัวเลขใน Excel

ปัญหา: StudentID เช่น 6000000001 ถูก Excel แสดงเป็น 6.00E+09
สาเหตุ: CSV บันทึก StudentID เป็น number แทน string

แก้ไข: เพิ่ม prefix "S" หรือ quotes เพื่อบังคับให้เป็น string
"""

import pandas as pd
from pathlib import Path

def fix_student_id_format(data_dir='synthetic_data'):
    """แก้ไข StudentID ให้เป็น string format"""

    data_dir = Path(data_dir)

    print("=" * 60)
    print("🔧 กำลังแก้ไข StudentID Format...")
    print("=" * 60)

    # รายชื่อไฟล์ที่มี StudentID
    files = [
        'ID_Group.csv',
        'SJ111777_Attendance15weeks.csv',
        'SJ111777_Score.csv',
        'SJ220777_Attendance15weeks.csv',
        'SJ220777_Score.csv',
        'SJ333777_Attendance15weeks.csv',
        'S333777_Score.csv'
    ]

    for filename in files:
        filepath = data_dir / filename

        if not filepath.exists():
            print(f"⚠️  ไม่พบไฟล์: {filename}")
            continue

        print(f"\n📝 กำลังแก้ไข: {filename}")

        # อ่านโดยบังคับให้ StudentID เป็น string
        df = pd.read_csv(filepath, dtype={'StudentID': str})

        # เพิ่ม prefix 'S' ถ้ายังไม่มี
        if not df['StudentID'].iloc[0].startswith('S'):
            df['StudentID'] = 'S' + df['StudentID'].astype(str)
            print(f"   ✓ เพิ่ม prefix 'S' แล้ว")
        else:
            print(f"   ✓ มี prefix 'S' อยู่แล้ว")

        # บันทึกกลับ (ด้วย quoting เพื่อให้ Excel อ่านถูก)
        df.to_csv(filepath, index=False, quoting=1)  # quoting=1 = QUOTE_ALL

        print(f"   ✓ บันทึกเรียบร้อย")
        print(f"   StudentID ตัวอย่าง: {df['StudentID'].iloc[0]}")

    print("\n" + "=" * 60)
    print("✅ แก้ไขเสร็จสมบูรณ์!")
    print("=" * 60)
    print("\nตอนนี้ StudentID จะเป็น:")
    print("  - ก่อน: 6000000001 (อาจแสดงเป็น 6.00E+09 ใน Excel)")
    print("  - หลัง: S6000000001 (แสดงเป็น text ใน Excel)")


def verify_fix(data_dir='synthetic_data'):
    """ตรวจสอบว่าแก้ไขถูกต้องแล้ว"""

    data_dir = Path(data_dir)

    print("\n" + "=" * 60)
    print("🔍 ตรวจสอบการแก้ไข...")
    print("=" * 60)

    # ตรวจสอบ ID_Group.csv
    df = pd.read_csv(data_dir / 'ID_Group.csv', dtype={'StudentID': str})

    print(f"\n📊 ID_Group.csv:")
    print(f"   จำนวน: {len(df)} แถว")
    print(f"   StudentID ตัวอย่าง 5 คนแรก:")

    for i in range(min(5, len(df))):
        sid = df['StudentID'].iloc[i]
        print(f"      {i+1}. {sid} (type: {type(sid).__name__}, len: {len(sid)})")

    # ตรวจสอบว่าทุก StudentID เริ่มด้วย 'S'
    all_start_with_s = df['StudentID'].str.startswith('S').all()

    if all_start_with_s:
        print("\n   ✅ ทุก StudentID เริ่มด้วย 'S' แล้ว")
    else:
        print("\n   ⚠️  ยังมี StudentID บางตัวไม่เริ่มด้วย 'S'")

    # ตรวจสอบความซ้ำซ้อน
    duplicates = df['StudentID'].duplicated().sum()
    print(f"   StudentID ซ้ำ: {duplicates} แถว")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    # แก้ไขไฟล์
    fix_student_id_format('synthetic_data')

    # ตรวจสอบ
    verify_fix('synthetic_data')

    print("\n💡 คำแนะนำ:")
    print("1. เปิดไฟล์ CSV ใน Excel ดูว่า StudentID แสดงเป็น 'S6000000001' แล้ว")
    print("2. ถ้ายังเป็นตัวเลข ให้เปิดด้วย Text Editor (VS Code, Notepad++) แทน")
    print("3. รันโปรเจคใหม่: python3 student_grade_prediction.py")
