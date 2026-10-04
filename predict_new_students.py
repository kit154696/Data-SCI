"""
Prediction Script for New Students
===================================
ใช้โมเดลที่เทรนแล้วเพื่อทำนายคะแนนนักศึกษาใหม่

วิธีใช้:
    python3 predict_new_students.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class StudentGradePredictor:
    """ทำนายเกรดนักศึกษา"""

    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.load_model()

    def load_model(self):
        """โหลดโมเดลและ scaler"""
        print("=" * 60)
        print("📦 กำลังโหลดโมเดล...")
        print("=" * 60)

        self.model = joblib.load(self.model_dir / 'best_model.pkl')
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')

        # โหลดรายชื่อฟีเจอร์
        with open(self.model_dir / 'feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f]

        print(f"✓ โหลดโมเดลสำเร็จ")
        print(f"✓ จำนวนฟีเจอร์: {len(self.feature_names)}")
        print()

    def predict(self, student_data):
        """
        ทำนายคะแนน

        Parameters:
        -----------
        student_data : dict or pd.DataFrame
            ข้อมูลนักศึกษา (ต้องมีฟีเจอร์ครบตาม feature_names)

        Returns:
        --------
        predictions : array
            คะแนนที่ทำนาย
        """
        if isinstance(student_data, dict):
            student_data = pd.DataFrame([student_data])

        # ตรวจสอบฟีเจอร์
        missing_features = set(self.feature_names) - set(student_data.columns)
        if missing_features:
            raise ValueError(f"ขาดฟีเจอร์: {missing_features}")

        # เรียงลำดับฟีเจอร์ให้ตรงกับตอนเทรน
        X = student_data[self.feature_names].copy()

        # จัดการค่าว่าง
        X = X.fillna(X.median())

        # Standardize
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)

        return predictions

    def predict_with_confidence(self, student_data):
        """ทำนายพร้อมระดับความเสี่ยง"""
        predictions = self.predict(student_data)

        results = []
        for i, score in enumerate(predictions):
            if score >= 80:
                risk_level = "ต่ำ"
                recommendation = "นักศึกษามีแนวโน้มได้เกรดดี ควรส่งเสริมต่อ"
            elif score >= 70:
                risk_level = "ปานกลาง"
                recommendation = "นักศึกษามีแนวโน้มได้เกรดปานกลาง ควรติดตามให้คำแนะนำ"
            elif score >= 60:
                risk_level = "สูง"
                recommendation = "นักศึกษามีแนวโน้มได้เกรดต่ำ ควรให้ความช่วยเหลือเร่งด่วน"
            else:
                risk_level = "สูงมาก"
                recommendation = "นักศึกษามีความเสี่ยงสูงมาก ต้องเข้าพบอาจารย์ทันที"

            results.append({
                'predicted_score': round(score, 2),
                'risk_level': risk_level,
                'recommendation': recommendation
            })

        return results


def demo_prediction():
    """ตัวอย่างการใช้งาน"""
    print("\n" + "=" * 60)
    print("  DEMO: ทำนายคะแนนนักศึกษาใหม่")
    print("=" * 60 + "\n")

    # สร้าง predictor
    predictor = StudentGradePredictor()

    # ตัวอย่างนักศึกษา 3 คน
    students = [
        {
            'name': 'นักศึกษา A (เก่ง)',
            'GPA': 3.8,
            'Group': 1,
            'sj111777_attendance_rate': 0.95,
            'sj111777_attendance_count': 14,
            'sj111777_late_attendance': 0,
            'sj111777_tasks_norm': 0.92,
            'sj111777_quiz_norm': 0.90,
            'sj111777_project_norm': 0.95,
            'sj111777_midterm_norm': 0.88,
            'sj111777_final_norm': 0.91,
            'sj111777_score': 91.5,
            'sj220777_attendance_rate': 0.93,
            'sj220777_attendance_count': 14,
            'sj220777_late_attendance': 0,
            'sj220777_tasks_norm': 0.89,
            'sj220777_quiz_norm': 0.87,
            'sj220777_project_norm': 0.91,
            'sj220777_midterm_norm': 0.86,
            'sj220777_final_norm': 0.88,
            'sj220777_score': 88.3,
            'avg_attendance_rate': 0.94,
            'min_attendance_rate': 0.93,
            'max_attendance_rate': 0.95,
            'total_avg_score': 89.9,
            'score_std': 1.6,
            'score_min': 88.3,
            'score_max': 91.5,
        },
        {
            'name': 'นักศึกษา B (ปานกลาง)',
            'GPA': 2.8,
            'Group': 2,
            'sj111777_attendance_rate': 0.80,
            'sj111777_attendance_count': 12,
            'sj111777_late_attendance': 1,
            'sj111777_tasks_norm': 0.75,
            'sj111777_quiz_norm': 0.72,
            'sj111777_project_norm': 0.78,
            'sj111777_midterm_norm': 0.70,
            'sj111777_final_norm': 0.73,
            'sj111777_score': 73.2,
            'sj220777_attendance_rate': 0.78,
            'sj220777_attendance_count': 12,
            'sj220777_late_attendance': 1,
            'sj220777_tasks_norm': 0.72,
            'sj220777_quiz_norm': 0.70,
            'sj220777_project_norm': 0.74,
            'sj220777_midterm_norm': 0.68,
            'sj220777_final_norm': 0.71,
            'sj220777_score': 70.8,
            'avg_attendance_rate': 0.79,
            'min_attendance_rate': 0.78,
            'max_attendance_rate': 0.80,
            'total_avg_score': 72.0,
            'score_std': 1.2,
            'score_min': 70.8,
            'score_max': 73.2,
        },
        {
            'name': 'นักศึกษา C (เสี่ยง)',
            'GPA': 2.2,
            'Group': 3,
            'sj111777_attendance_rate': 0.60,
            'sj111777_attendance_count': 9,
            'sj111777_late_attendance': 1,
            'sj111777_tasks_norm': 0.55,
            'sj111777_quiz_norm': 0.52,
            'sj111777_project_norm': 0.58,
            'sj111777_midterm_norm': 0.50,
            'sj111777_final_norm': 0.53,
            'sj111777_score': 53.5,
            'sj220777_attendance_rate': 0.57,
            'sj220777_attendance_count': 8,
            'sj220777_late_attendance': 1,
            'sj220777_tasks_norm': 0.52,
            'sj220777_quiz_norm': 0.48,
            'sj220777_project_norm': 0.54,
            'sj220777_midterm_norm': 0.47,
            'sj220777_final_norm': 0.50,
            'sj220777_score': 50.2,
            'avg_attendance_rate': 0.585,
            'min_attendance_rate': 0.57,
            'max_attendance_rate': 0.60,
            'total_avg_score': 51.85,
            'score_std': 1.65,
            'score_min': 50.2,
            'score_max': 53.5,
        }
    ]

    # ทำนายแต่ละคน
    for student in students:
        name = student.pop('name')
        print(f"\n{'=' * 60}")
        print(f"📊 {name}")
        print(f"{'=' * 60}")
        print(f"GPA เดิม: {student['GPA']:.2f}")
        print(f"คะแนนเฉลี่ยในอดีต: {student['total_avg_score']:.2f}")
        print(f"อัตราการเข้าเรียนเฉลี่ย: {student['avg_attendance_rate']*100:.1f}%")

        # ทำนาย
        results = predictor.predict_with_confidence(student)
        result = results[0]

        print(f"\n🎯 คะแนนที่คาดการณ์: {result['predicted_score']:.2f}")
        print(f"⚠️  ระดับความเสี่ยง: {result['risk_level']}")
        print(f"💡 คำแนะนำ: {result['recommendation']}")

    print("\n" + "=" * 60)
    print("✅ เสร็จสิ้นการทำนาย")
    print("=" * 60)


if __name__ == '__main__':
    demo_prediction()
