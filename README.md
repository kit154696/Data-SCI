# Student Grade Prediction Project
## ทำนายผลการเรียนของนักศึกษาด้วย Machine Learning

โครงงานนี้ใช้ Machine Learning เพื่อทำนายผลการเรียนของนักศึกษาในเทอมถัดไป โดยอิงจากข้อมูลการเข้าเรียนและผลการเรียนในอดีต

---

## 🎯 วัตถุประสงค์

**Business Question**: ผลการเรียนของนักศึกษาเป็นอย่างไรบ้าง

**Data Science Question**: เราสามารถใช้ข้อมูลในอดีตและพฤติกรรมการเรียนของนักศึกษา เพื่อสร้างโมเดล ML สำหรับทำนายเกรดที่นักศึกษาน่าจะได้รับในเทอมหน้า ได้หรือไม่

---

## 📊 ข้อมูล

### Dataset
ข้อมูลการเข้าเรียนและผลการเรียนของนักศึกษา จาก **3 รายวิชา**:
- SJ111777
- SJ220777  
- SJ333777 (Target - วิชาที่ต้องการทำนาย)

### ไฟล์ข้อมูล (7 ตาราง)
```
synthetic_data/
├── ID_Group.csv                        # ข้อมูลพื้นฐาน: StudentID, Group, GPA
├── SJ111777_Attendance15weeks.csv      # การเข้าเรียน 15 สัปดาห์
├── SJ111777_Score.csv                  # คะแนนต่างๆ
├── SJ220777_Attendance15weeks.csv      # การเข้าเรียน 15 สัปดาห์
├── SJ220777_Score.csv                  # คะแนนต่างๆ
├── SJ333777_Attendance15weeks.csv      # การเข้าเรียน 15 สัปดาห์
└── S333777_Score.csv                   # คะแนนต่างๆ (Target)
```

---

## 🚀 Quick Start

### 1. ติดตั้ง Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### 2. สร้างข้อมูล Synthetic

```bash
python3 data-cr-test.py
```

ผลลัพธ์: โฟลเดอร์ `synthetic_data/` มีไฟล์ CSV 7 ไฟล์

### 3. รัน Pipeline หลัก

```bash
python3 student_grade_prediction.py
```

ผลลัพธ์:
- `analysis_output/` - กราฟและการวิเคราะห์ (6 ไฟล์ PNG)
- `models/` - โมเดลที่เทรนแล้ว (best_model.pkl, scaler.pkl)

### 4. ทดสอบการทำนาย

```bash
python3 predict_new_students.py
```

ผลลัพธ์: ทำนายคะแนนนักศึกษา 3 คน (Demo)

### 5. สร้างรายงานนำเสนอ

```bash
python3 create_presentation_report.py
```

ผลลัพธ์: `analysis_output/PRESENTATION_REPORT.md` - รายงานฉบับสมบูรณ์

---

## 📁 โครงสร้างโปรเจค

```
datasci/
├── README.md                           # ไฟล์นี้
├── data-cr-test.py                     # สร้างข้อมูล synthetic
├── student_grade_prediction.py         # Pipeline หลัก (EDA + Modeling)
├── predict_new_students.py             # ทำนายนักศึกษาใหม่
├── create_presentation_report.py       # สร้างรายงาน
│
├── synthetic_data/                     # ข้อมูล (7 ไฟล์)
│   ├── ID_Group.csv
│   ├── SJ111777_Attendance15weeks.csv
│   ├── SJ111777_Score.csv
│   ├── SJ220777_Attendance15weeks.csv
│   ├── SJ220777_Score.csv
│   ├── S333777_Score.csv
│   └── SJ333777_Attendance15weeks.csv
│
├── analysis_output/                    # กราฟและรายงาน
│   ├── target_distribution.png
│   ├── correlation_matrix.png
│   ├── feature_correlation.png
│   ├── gpa_vs_target.png
│   ├── attendance_vs_target.png
│   ├── model_comparison.png
│   └── PRESENTATION_REPORT.md
│
└── models/                             # โมเดลที่บันทึก
    ├── best_model.pkl
    ├── scaler.pkl
    └── feature_names.txt
```

---

## 🔬 Methodology

### 1. Data Cleaning
- ลบแถวซ้ำ
- เติมค่าว่างด้วย median
- จัดการนักศึกษาที่ไม่มีข้อมูลบางวิชา

### 2. Feature Engineering (27 features)
- **Attendance Features**: อัตราการเข้าเรียน, จำนวนครั้ง, พฤติกรรมการเข้าเรียน
- **Score Features**: คะแนนแต่ละส่วน (normalized)
- **Aggregated Features**: ค่าเฉลี่ย, min, max, std ของคะแนนและการเข้าเรียน
- **Baseline Features**: GPA, Group

### 3. Models Tested
- Linear Regression
- Ridge Regression
- **Lasso Regression** ⭐ (Best)
- Random Forest
- Gradient Boosting

### 4. Evaluation
- Train/Test Split: 80/20
- Cross-Validation: 5-fold
- Metrics: R², RMSE, MAE

---

## 📈 ผลลัพธ์

### Model Performance

| Metric | Value |
|--------|-------|
| **Test R²** | **0.9587** (อธิบายความแปรปรวน 95.87%) |
| **RMSE** | **2.24** คะแนน |
| **MAE** | **1.72** คะแนน |
| **CV R²** | **0.9535** (±0.011) |

### Best Model: Lasso Regression

**เหตุผล**:
- ✅ Test R² สูงสุด (0.9587)
- ✅ RMSE ต่ำสุด (2.24)
- ✅ ไม่ Overfit
- ✅ Cross-validation สูงและเสถียร
- ✅ Automatic feature selection

---

## 💡 Key Insights

1. **GPA เดิม** มีความสัมพันธ์สูงกับคะแนนในเทอมหน้า (r ≈ 0.85)

2. **คะแนนวิชาก่อนหน้า** เป็นตัวทำนายที่แม่นยำที่สุด (r ≈ 0.92)

3. **การเข้าเรียน** มีผลต่อคะแนน (r ≈ 0.65)
   - นักศึกษาที่เข้าเรียน > 80% มักได้คะแนนดีกว่า

4. **ความสม่ำเสมอ** มีความสำคัญ
   - นักศึกษาที่มีคะแนนสม่ำเสมอมีแนวโน้มดีกว่า

### ✅ Hypothesis Validation

**ยืนยันสมมติฐาน**: พฤติกรรมการเรียนในอดีต (GPA, คะแนน, การเข้าเรียน) สามารถทำนายผลการเรียนในอนาคตได้อย่างแม่นยำ (R² > 0.95)

---

## 🎯 Business Recommendations

### 1. Early Warning System
สร้างระบบแจ้งเตือนล่วงหน้าเพื่อระบุกลุ่มเสี่ยง

### 2. Targeted Intervention
มาตรการช่วยเหลือแบบเจาะจงตามระดับความเสี่ยง:

| Predicted Score | Risk Level | Action |
|----------------|------------|--------|
| ≥ 80 | ต่ำ | ส่งเสริมต่อ |
| 70-79 | ปานกลาง | ติดตามให้คำแนะนำ |
| 60-69 | สูง | ให้ความช่วยเหลือเร่งด่วน |
| < 60 | สูงมาก | เข้าพบอาจารย์ทันที |

### 3. Attendance Monitoring
เพิ่มความสำคัญของการเข้าเรียน และแจ้งเตือนเมื่อขาดเรียนบ่อย

### 4. Continuous Improvement
วิเคราะห์และปรับปรุงการสอนอย่างต่อเนื่อง

---

## 🔮 Next Steps

### Phase 1: Deployment (1-2 เดือน)
- [ ] ทดสอบโมเดลกับข้อมูลจริง
- [ ] สร้าง Web API สำหรับทำนาย real-time
- [ ] พัฒนา Dashboard สำหรับอาจารย์
- [ ] จัดทำคู่มือการใช้งาน

### Phase 2: Enhancement (3-6 เดือน)
- [ ] รวบรวมข้อมูลเพิ่มเติม (ประชากรศาสตร์, การมีส่วนร่วม)
- [ ] ทดลองโมเดลขั้นสูง (Neural Networks, XGBoost)
- [ ] Retrain โมเดลทุก semester
- [ ] ขยายไปวิชาอื่นๆ

### Phase 3: Integration (6-12 เดือน)
- [ ] เชื่อมต่อกับระบบทะเบียน
- [ ] พัฒนา Mobile App
- [ ] ระบบแจ้งเตือนอัตโนมัติ
- [ ] ทำนายความเสี่ยงการออกกลางคัน (Dropout)

---

## 🛠️ Technical Stack

- **Python 3.x**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Model Persistence**: joblib

---

## 📊 Sample Usage

### ทำนายคะแนนนักศึกษาใหม่

```python
from predict_new_students import StudentGradePredictor

# โหลดโมเดล
predictor = StudentGradePredictor()

# ข้อมูลนักศึกษา
student_data = {
    'GPA': 3.5,
    'sj111777_score': 85.0,
    'sj220777_score': 82.0,
    'avg_attendance_rate': 0.90,
    # ... (27 features ทั้งหมด)
}

# ทำนาย
results = predictor.predict_with_confidence(student_data)
print(f"คะแนนที่คาดการณ์: {results[0]['predicted_score']:.2f}")
print(f"ระดับความเสี่ยง: {results[0]['risk_level']}")
print(f"คำแนะนำ: {results[0]['recommendation']}")
```

---

## 📖 Documentation

รายละเอียดเพิ่มเติมใน:
- **PRESENTATION_REPORT.md** - รายงานฉบับเต็ม (20+ หน้า)
- **analysis_output/** - กราฟและการวิเคราะห์ทั้งหมด

---

## 📝 License

This project is created for educational purposes.

---

## 🙏 Acknowledgments

- Dataset: Student Learning Analytics (Synthetic Data)
- Tools: Python, scikit-learn, pandas
- Visualization: matplotlib, seaborn

---

## 📧 Contact

สำหรับคำถามและข้อเสนอแนะ:
- Project: Student Grade Prediction
- Date: 2026-10-03

---

**สร้างด้วย ❤️ และ Python**
# Data-SCI
