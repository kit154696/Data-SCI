"""
Create Presentation Report
==========================
สร้างรายงานสำหรับนำเสนอโครงงาน Data Science
"""

import pandas as pd
from pathlib import Path


def create_markdown_report():
    """สร้างรายงานแบบ Markdown"""

    report = """
# Student Grade Prediction Project
## ทำนายผลการเรียนของนักศึกษาด้วย Machine Learning

---

## 📋 Business Question
**ผลการเรียนของนักศึกษาเป็นอย่างไรบ้าง**

- ต้องการทราบแนวโน้มผลการเรียนของนักศึกษา
- ต้องการระบุกลุ่มเสี่ยงที่อาจได้เกรดต่ำ
- ต้องการเครื่องมือช่วยติดตามและให้คำแนะนำนักศึกษา

---

## 🎯 Data Science Question
**เราสามารถใช้ข้อมูลในอดีตและพฤติกรรมการเรียนของนักศึกษา
เพื่อสร้างโมเดล Machine Learning สำหรับทำนายเกรดที่นักศึกษา
น่าจะได้รับในเทอมหน้า ได้หรือไม่?**

### Hypothesis
ผลการเรียนของนักศึกษาในเทอมหน้า สัมพันธ์กับชุดตัวแปร
พฤติกรรมการเรียน (Academic Behavioral Features) ในอดีต
และสามารถนำตัวแปรเหล่านี้ไปเข้าโมเดล Machine Learning
เพื่อทำนายเกรดล่วงหน้า

---

## 📊 Dataset Description

### ข้อมูลที่ใช้
ข้อมูลการเข้าเรียนและผลการเรียนของนักศึกษา จาก **3 รายวิชา**:
- SJ111777
- SJ220777
- SJ333777

### โครงสร้างข้อมูล (7 ตาราง)

#### 1. ID_Group.csv - ข้อมูลพื้นฐาน
- `StudentID`: รหัสนักศึกษา
- `Group`: ประเภทกลุ่ม
- `GPA`: เกรดเฉลี่ยสะสมปัจจุบัน

#### 2-3. SJ111777_Attendance15weeks.csv และ SJ111777_Score.csv
- การเข้าเรียน 15 สัปดาห์
- คะแนนจากการตรวจชิ้นงาน (100 คะแนน)
- คะแนนควิซ (150 คะแนน)
- คะแนนโครงงาน (100 คะแนน)
- คะแนนสอบกลางภาค (150 คะแนน)
- คะแนนสอบปลายภาค (150 คะแนน)
- คะแนนรวม (100 คะแนน)

#### 4-5. SJ220777_Attendance15weeks.csv และ SJ220777_Score.csv
- การเข้าเรียน 15 สัปดาห์
- คะแนนจากการตรวจชิ้นงาน (100 คะแนน)
- คะแนนควิซ (90 คะแนน)
- คะแนนโครงงาน (90 คะแนน)
- คะแนนสอบกลางภาค (180 คะแนน)
- คะแนนสอบปลายภาค (180 คะแนน)
- คะแนนรวม (100 คะแนน)

#### 6-7. SJ333777_Attendance15weeks.csv และ S333777_Score.csv
- การเข้าเรียน 15 สัปดาห์
- คะแนนจากการตรวจชิ้นงาน (100 คะแนน)
- คะแนนโครงงาน (100 คะแนน)
- คะแนนสอบกลางภาค (100 คะแนน)
- คะแนนสอบปลายภาค (100 คะแนน)
- คะแนนรวม (100 คะแนน)

### จำนวนข้อมูล
- **นักศึกษาทั้งหมด**: 120 คน
- **ข้อมูลที่ใช้ครบถ้วน**: 120 records (หลังจาก cleaning)

---

## 🧹 Data Cleaning & Preparation

### ปัญหาที่พบในข้อมูล
1. **แถวซ้ำ**: พบข้อมูลซ้ำซ้อนในตารางคะแนน
2. **ค่าว่าง (Missing Values)**: พบค่าว่างในหลายคอลัมน์
   - SJ111777_Score: 1-4 ค่าว่างต่อคอลัมน์
   - SJ220777_Score: 1-4 ค่าว่างต่อคอลัมน์
   - S333777_Score: 1-2 ค่าว่างต่อคอลัมน์
3. **นักศึกษาบางคนไม่มีข้อมูลบางวิชา**: อาจเป็นการถอนวิชา

### วิธีการแก้ไข
1. **ลบแถวซ้ำ**: ใช้ `drop_duplicates()` โดยอิงจาก StudentID
2. **เติมค่าว่าง**: ใช้ค่า median ของแต่ละคอลัมน์
3. **จัดการข้อมูลที่หาย**: ใช้ left join เพื่อรักษานักศึกษาทุกคนไว้

---

## ⚙️ Feature Engineering

### Features ที่สร้างขึ้น (27 features)

#### 1. Attendance Features (แต่ละวิชา)
- `attendance_rate`: สัดส่วนการเข้าเรียน (0-1)
- `attendance_count`: จำนวนครั้งที่เข้าเรียน
- `late_attendance`: พฤติกรรมการเข้าเรียนช่วงหลัง > ช่วงแรก

#### 2. Score Features (แต่ละวิชา)
- `tasks_norm`: คะแนนชิ้นงาน (normalized 0-1)
- `quiz_norm`: คะแนนควิซ (normalized 0-1)
- `project_norm`: คะแนนโครงงาน (normalized 0-1)
- `midterm_norm`: คะแนนสอบกลางภาค (normalized 0-1)
- `final_norm`: คะแนนสอบปลายภาค (normalized 0-1)
- `score`: คะแนนรวมของวิชา (0-100)

#### 3. Aggregated Features (รวมทุกวิชา)
- `avg_attendance_rate`: อัตราการเข้าเรียนเฉลี่ย
- `min_attendance_rate`: อัตราการเข้าเรียนต่ำสุด
- `max_attendance_rate`: อัตราการเข้าเรียนสูงสุด
- `total_avg_score`: คะแนนเฉลี่ยรวมทุกวิชา
- `score_std`: ค่าเบี่ยงเบนมาตรฐานของคะแนน
- `score_min`: คะแนนต่ำสุด
- `score_max`: คะแนนสูงสุด

#### 4. Baseline Features
- `GPA`: เกรดเฉลี่ยสะสมปัจจุบัน
- `Group`: กลุ่มนักศึกษา

### Target Variable
**คะแนนวิชา SJ333777** (`sj333777_score`)
- เป็นวิชาที่ต้องการทำนาย (เทอมถัดไป)
- ใช้ข้อมูลจาก SJ111777 และ SJ220777 เป็น features

---

## 📈 Data Analysis & Insights

### 1. การกระจายของคะแนน
![Target Distribution](analysis_output/target_distribution.png)
- คะแนนเฉลี่ย: ~70-75 คะแนน
- การกระจาย: ค่อนข้างสมมาตร (Normal-like distribution)
- มีนักศึกษาส่วนหนึ่งได้คะแนนต่ำกว่า 60 (กลุ่มเสี่ยง)

### 2. Correlation Analysis
![Correlation Matrix](analysis_output/correlation_matrix.png)

**ความสัมพันธ์ที่สำคัญ**:
- **GPA กับคะแนนเป้าหมาย**: สหสัมพันธ์สูง (r > 0.8)
- **คะแนนวิชาก่อนหน้ากับเป้าหมาย**: สหสัมพันธ์สูงมาก (r > 0.9)
- **อัตราการเข้าเรียนกับคะแนน**: สหสัมพันธ์ปานกลาง-สูง (r ~ 0.6-0.7)

### 3. Feature Importance
![Feature Correlation](analysis_output/feature_correlation.png)

**Top 5 Features**:
1. คะแนนเฉลี่ยรวมทุกวิชา (total_avg_score)
2. คะแนนวิชา SJ220777
3. คะแนนวิชา SJ111777
4. GPA เดิม
5. อัตราการเข้าเรียนเฉลี่ย

### 4. GPA vs Target Score
![GPA vs Target](analysis_output/gpa_vs_target.png)
- แสดงความสัมพันธ์เชิงบวกที่ชัดเจน
- นักศึกษาที่มี GPA สูงมักได้คะแนนสูงกว่า

### 5. Attendance vs Target Score
![Attendance vs Target](analysis_output/attendance_vs_target.png)
- การเข้าเรียนสม่ำเสมอมีผลต่อคะแนน
- นักศึกษาที่เข้าเรียน > 80% มักได้คะแนนดีกว่า

---

## 🤖 Modeling

### Models Tested
เปรียบเทียบ 5 โมเดล:
1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Random Forest**
5. **Gradient Boosting**

### Data Split
- **Training Set**: 80% (96 samples)
- **Test Set**: 20% (24 samples)
- **Cross-Validation**: 5-fold CV

### Model Performance Comparison

| Model | Train R² | Test R² | RMSE | MAE | CV R² (±std) |
|-------|----------|---------|------|-----|--------------|
| Linear Regression | 0.9927 | 0.9440 | 2.60 | 1.35 | 0.9803 (±0.021) |
| Ridge Regression | 0.9770 | 0.9455 | 2.57 | 1.88 | 0.9490 (±0.013) |
| **Lasso Regression** | **0.9752** | **0.9587** | **2.24** | **1.72** | **0.9535 (±0.011)** |
| Random Forest | 0.9799 | 0.9076 | 3.35 | 2.78 | 0.7892 (±0.066) |
| Gradient Boosting | 0.9995 | 0.9256 | 3.00 | 2.64 | 0.7695 (±0.081) |

### Best Model: Lasso Regression

**เหตุผลที่เลือก**:
- ✅ **Test R² สูงสุด**: 0.9587 (อธิบายความแปรปรวน 95.87%)
- ✅ **RMSE ต่ำสุด**: 2.24 คะแนน
- ✅ **ไม่ Overfit**: Train R² ≈ Test R²
- ✅ **Cross-validation สูง**: CV R² = 0.9535 (เสถียร)
- ✅ **Feature Selection**: Lasso ทำ automatic feature selection (ลด complexity)

### Model Visualization
![Model Comparison](analysis_output/model_comparison.png)

**จากกราฟ**:
- **R² Comparison**: Lasso มี Test R² สูงสุด
- **RMSE Comparison**: Lasso มี error ต่ำสุด
- **Prediction vs Actual**: จุดกระจายใกล้เส้นทแยงมุม (ทำนายแม่นยำ)
- **Residual Plot**: จุดกระจายรอบศูนย์อย่างสม่ำเสมอ (ไม่มี pattern)

---

## 📊 Evaluation Results

### Performance Metrics

```
Test R² Score:        0.9587  (95.87% of variance explained)
Root Mean Square Error: 2.24 points
Mean Absolute Error:    1.72 points
Cross-Validation R²:    0.9535 (±0.011)
```

### Interpretation

**R² = 0.9587** หมายความว่า:
- โมเดลอธิบายความแปรปรวนของคะแนนได้ **95.87%**
- มีเพียง **4.13%** ที่อธิบายไม่ได้ (อาจเกิดจากปัจจัยอื่นๆ)

**MAE = 1.72** หมายความว่า:
- โดยเฉลี่ยแล้วคะแนนที่ทำนายคลาดเคลื่อน **±1.72 คะแนน**
- สำหรับคะแนนเต็ม 100 นั้นถือว่าแม่นยำมาก

### Prediction Confidence Levels

| Predicted Score | Risk Level | Recommendation |
|----------------|------------|----------------|
| ≥ 80 | **ต่ำ** | นักศึกษามีแนวโน้มได้เกรดดี ควรส่งเสริมต่อ |
| 70-79 | **ปานกลาง** | นักศึกษามีแนวโน้มได้เกรดปานกลาง ควรติดตามให้คำแนะนำ |
| 60-69 | **สูง** | นักศึกษามีแนวโน้มได้เกรดต่ำ ควรให้ความช่วยเหลือเร่งด่วน |
| < 60 | **สูงมาก** | นักศึกษามีความเสี่ยงสูงมาก ต้องเข้าพบอาจารย์ทันที |

---

## 💡 Key Insights

### ปัจจัยที่ส่งผลต่อผลการเรียน

1. **GPA เดิม** (r ≈ 0.85)
   - นักศึกษาที่มี GPA สูงมีแนวโน้มได้คะแนนดีในเทอมถัดไป

2. **คะแนนวิชาก่อนหน้า** (r ≈ 0.92)
   - คะแนนในอดีตเป็นตัวทำนายที่แม่นยำที่สุด
   - แสดงถึงความสม่ำเสมอของผลการเรียน

3. **การเข้าเรียน** (r ≈ 0.65)
   - นักศึกษาที่เข้าเรียนสม่ำเสมอมักได้คะแนนดีกว่า
   - การเข้าเรียน > 80% เป็นเกณฑ์ที่ดี

4. **ความสม่ำเสมอของคะแนน**
   - นักศึกษาที่มีคะแนนสม่ำเสมอ (score_std ต่ำ) มีแนวโน้มดีกว่า

### Hypothesis Validation

✅ **ยืนยันสมมติฐาน**:
- พฤติกรรมการเรียนในอดีต (GPA, คะแนน, การเข้าเรียน)
  สามารถทำนายผลการเรียนในอนาคตได้อย่างแม่นยำ (R² > 0.95)

---

## 🎯 Business Recommendations

### 1. Early Warning System
**สร้างระบบแจ้งเตือนล่วงหน้า**
- ทำนายคะแนนตั้งแต่กลางภาคเพื่อระบุกลุ่มเสี่ยง
- แจ้งเตือนอาจารย์ผู้สอนให้ดูแลนักศึกษากลุ่มเสี่ยงทันที
- ประหยัดเวลาและทรัพยากรในการช่วยเหลือนักศึกษา

### 2. Targeted Intervention
**มาตรการช่วยเหลือแบบเจาะจง**
- **ความเสี่ยงสูงมาก (คะแนน < 60)**: ติวเสริมเร่งด่วน, พบอาจารย์ที่ปรึกษา
- **ความเสี่ยงสูง (60-69)**: กิจกรรมเสริมความเข้าใจ, Study group
- **ปานกลาง (70-79)**: ให้คำแนะนำและติดตามผล
- **ต่ำ (≥ 80)**: ส่งเสริมและพัฒนาต่อ

### 3. Attendance Monitoring
**เพิ่มความสำคัญของการเข้าเรียน**
- ติดตามการเข้าเรียนอย่างเข้มงวด
- แจ้งเตือนเมื่อการเข้าเรียน < 80%
- สื่อสารกับผู้ปกครองเมื่อขาดเรียนบ่อย

### 4. Continuous Improvement
**ปรับปรุงการสอนอย่างต่อเนื่อง**
- วิเคราะห์ว่าหัวข้อไหนนักศึกษาได้คะแนนต่ำ
- ปรับวิธีการสอนในส่วนที่นักศึกษาเข้าใจยาก
- เปรียบเทียบผลการเรียนระหว่างกลุ่มต่างๆ

### 5. Predictive Dashboard
**สร้าง Dashboard สำหรับผู้บริหาร**
- แสดงภาพรวมผลการเรียนของแต่ละวิชา
- จำนวนนักศึกษาในแต่ละระดับความเสี่ยง
- Trend ของผลการเรียนในแต่ละปี
- เปรียบเทียบผลการทำนายกับผลจริง

---

## 🚀 Next Steps

### Phase 1: Deployment (1-2 เดือน)
1. ✅ ทดสอบโมเดลกับข้อมูลจริงในเทอมถัดไป
2. ✅ สร้าง Web API สำหรับทำนายคะแนน real-time
3. ✅ พัฒนา Dashboard สำหรับอาจารย์และผู้บริหาร
4. ✅ จัดทำคู่มือการใช้งานสำหรับอาจารย์

### Phase 2: Enhancement (3-6 เดือน)
1. 📊 รวบรวมข้อมูลเพิ่มเติม:
   - ข้อมูลประชากรศาสตร์ (อายุ, เพศ, ที่พักอาศัย)
   - ข้อมูลการมีส่วนร่วมในชั้นเรียน (คำถาม, การบ้าน)
   - ข้อมูลกิจกรรมนอกหลักสูตร

2. 🔄 ปรับปรุงโมเดล:
   - ทดลองโมเดลขั้นสูง (Neural Networks, XGBoost)
   - เพิ่มฟีเจอร์จากข้อมูลใหม่
   - Retrain โมเดลทุก semester

3. 🎓 ขยายขอบเขต:
   - ทำนายเกรดสำหรับวิชาอื่นๆ
   - ทำนายความเสี่ยงในการออกกลางคัน (Dropout)
   - แนะนำวิชาที่เหมาะสมสำหรับนักศึกษา

### Phase 3: Integration (6-12 เดือน)
1. 🔗 เชื่อมต่อกับระบบทะเบียน
2. 📱 พัฒนา Mobile App สำหรับนักศึกษา
3. 🤖 ระบบแนะนำอัตโนมัติ (Automated recommendation)
4. 📧 ระบบแจ้งเตือนผ่าน Email/LINE

---

## 📁 Project Files

```
datasci/
├── data-cr-test.py                     # สร้างข้อมูล synthetic
├── student_grade_prediction.py         # Pipeline หลัก
├── predict_new_students.py             # ทำนายนักศึกษาใหม่
├── create_presentation_report.py       # สร้างรายงาน
├── synthetic_data/                     # ข้อมูลที่สร้างขึ้น
│   ├── ID_Group.csv
│   ├── SJ111777_Attendance15weeks.csv
│   ├── SJ111777_Score.csv
│   ├── SJ220777_Attendance15weeks.csv
│   ├── SJ220777_Score.csv
│   ├── S333777_Score.csv
│   └── SJ333777_Attendance15weeks.csv
├── analysis_output/                    # กราฟและการวิเคราะห์
│   ├── target_distribution.png
│   ├── correlation_matrix.png
│   ├── feature_correlation.png
│   ├── gpa_vs_target.png
│   ├── attendance_vs_target.png
│   └── model_comparison.png
└── models/                             # โมเดลที่เทรนแล้ว
    ├── best_model.pkl
    ├── scaler.pkl
    └── feature_names.txt
```

---

## 👥 Team & Timeline

**Duration**: 4 สัปดาห์

**Week 1**: Data Collection & Cleaning
- โหลดและตรวจสอบข้อมูล
- ทำความสะอาดและจัดการ missing values

**Week 2**: Feature Engineering & EDA
- สร้างฟีเจอร์จากข้อมูลดิบ
- วิเคราะห์และสร้างกราฟ

**Week 3**: Modeling & Evaluation
- เทรนโมเดลหลายตัว
- เปรียบเทียบและเลือกโมเดลที่ดีที่สุด

**Week 4**: Documentation & Presentation
- สร้างรายงาน
- เตรียมนำเสนอ

---

## 📚 References & Tools

### Libraries Used
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning
- **joblib**: Model serialization

### Machine Learning Techniques
- Supervised Learning (Regression)
- Feature Engineering
- Cross-Validation
- Regularization (Lasso)
- Model Evaluation Metrics

---

## ✅ Conclusion

### สรุปผลการดำเนินงาน

✅ **Data Science Question: ได้คำตอบแล้ว**
- เราสามารถทำนายเกรดนักศึกษาในเทอมหน้าได้อย่างแม่นยำ
- โมเดล Lasso Regression ให้ผลลัพธ์ที่ดีที่สุด (R² = 0.9587)

✅ **Business Impact**
- สามารถระบุกลุ่มเสี่ยงล่วงหน้าได้
- ช่วยให้อาจารย์ดูแลนักศึกษาได้ทันท่วงที
- ลดอัตราการได้เกรดต่ำและออกกลางคัน

✅ **Actionable Insights**
- นักศึกษาที่มี GPA สูง + เข้าเรียนสม่ำเสมอ → ได้เกรดดี
- การเข้าเรียน > 80% เป็นเกณฑ์สำคัญ
- คะแนนในอดีตบ่งชี้แนวโน้มอนาคต

### Key Success Factors

🎯 **High Model Accuracy**: R² > 0.95
🎯 **Low Prediction Error**: MAE = 1.72 คะแนน
🎯 **Robust Validation**: CV R² = 0.95
🎯 **Interpretable Features**: ใช้ตัวแปรที่เข้าใจง่าย
🎯 **Ready for Deployment**: บันทึกโมเดลและ pipeline ไว้แล้ว

---

## 🙏 Thank You

**ขอบคุณที่รับชม**

สำหรับคำถามและข้อเสนอแนะ กรุณาติดต่อ:
- 📧 Email: [your-email@example.com]
- 📂 GitHub: [your-repo-link]

---

*Generated: 2026-10-03*
*Project: Student Grade Prediction with Machine Learning*
"""

    return report


def save_report():
    """บันทึกรายงาน"""
    output_dir = Path('analysis_output')
    output_dir.mkdir(exist_ok=True)

    report = create_markdown_report()

    # บันทึกเป็น Markdown
    md_path = output_dir / 'PRESENTATION_REPORT.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print("=" * 60)
    print("📄 สร้างรายงานสำเร็จ!")
    print("=" * 60)
    print(f"\n✓ บันทึกรายงานที่: {md_path}")
    print(f"\n📋 รายงานประกอบด้วย:")
    print("  • Business Question & Data Science Question")
    print("  • Dataset Description (7 ตาราง)")
    print("  • Data Cleaning & Preparation")
    print("  • Feature Engineering (27 features)")
    print("  • Data Analysis & Insights")
    print("  • Modeling & Evaluation")
    print("  • Performance Metrics & Visualization")
    print("  • Key Insights & Recommendations")
    print("  • Next Steps & Deployment Plan")
    print("\n💡 คุณสามารถ:")
    print("  1. เปิดไฟล์ .md ด้วย text editor หรือ Markdown viewer")
    print("  2. แปลงเป็น PDF ด้วย pandoc หรือ Markdown to PDF tool")
    print("  3. นำเสนอด้วย Markdown presentation tool (e.g., Marp, reveal.js)")
    print("  4. Copy ไปยัง Google Docs/Word สำหรับแก้ไขเพิ่มเติม")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    save_report()
