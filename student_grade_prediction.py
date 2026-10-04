"""
Student Grade Prediction Pipeline
==================================
โครงงาน: ทำนายผลการเรียนของนักศึกษาในเทอมถัดไป

Business Question: ผลการเรียนของนักศึกษาเป็นอย่างไรบ้าง

Data Science Question:
เราสามารถใช้ข้อมูลในอดีตและพฤติกรรมการเรียนของนักศึกษา
เพื่อสร้างโมเดล ML สำหรับทำนายเกรดที่นักศึกษาน่าจะได้รับในเทอมหน้า ได้หรือไม่

Hypothesis:
ผลการเรียนของนักศึกษาในเทอมหน้า สัมพันธ์กับชุดตัวแปรพฤติกรรมการเรียน
(Academic Behavioral Features) ในอดีต และสามารถนำตัวแปรเหล่านี้ไปเข้าโมเดล
Machine Learning เพื่อทำนายเกรดล่วงหน้า
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

warnings.filterwarnings('ignore')

# Setup English display (default fonts)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== 1. DATA LOADING ====================
class DataLoader:
    """โหลดและรวมข้อมูลจากทั้ง 7 ตาราง"""

    def __init__(self, data_dir='synthetic_data'):
        self.data_dir = Path(data_dir)

    def load_all(self):
        """โหลดข้อมูลทั้งหมด"""
        print("=" * 60)
        print("📂 กำลังโหลดข้อมูล...")
        print("=" * 60)

        # โหลดตารางหลัก
        self.id_group = pd.read_csv(self.data_dir / 'ID_Group.csv')
        print(f"✓ ID_Group: {self.id_group.shape}")

        # วิชา SJ111777
        self.sj111_att = pd.read_csv(self.data_dir / 'SJ111777_Attendance15weeks.csv')
        self.sj111_score = pd.read_csv(self.data_dir / 'SJ111777_Score.csv')
        print(f"✓ SJ111777 - Attendance: {self.sj111_att.shape}, Score: {self.sj111_score.shape}")

        # วิชา SJ220777
        self.sj220_att = pd.read_csv(self.data_dir / 'SJ220777_Attendance15weeks.csv')
        self.sj220_score = pd.read_csv(self.data_dir / 'SJ220777_Score.csv')
        print(f"✓ SJ220777 - Attendance: {self.sj220_att.shape}, Score: {self.sj220_score.shape}")

        # วิชา SJ333777
        self.sj333_att = pd.read_csv(self.data_dir / 'SJ333777_Attendance15weeks.csv')
        self.s333_score = pd.read_csv(self.data_dir / 'S333777_Score.csv')
        print(f"✓ SJ333777 - Attendance: {self.sj333_att.shape}, Score: {self.s333_score.shape}")

        return self


# ==================== 2. DATA CLEANING & PREPARATION ====================
class DataCleaner:
    """ทำความสะอาดข้อมูล"""

    def __init__(self, loader):
        self.loader = loader

    def clean(self):
        """ทำความสะอาดข้อมูลทั้งหมด"""
        print("\n" + "=" * 60)
        print("🧹 กำลังทำความสะอาดข้อมูล...")
        print("=" * 60)

        # ลบแถวซ้ำ
        self.loader.sj111_score = self.loader.sj111_score.drop_duplicates(subset=['StudentID'])
        self.loader.sj220_score = self.loader.sj220_score.drop_duplicates(subset=['StudentID'])
        self.loader.s333_score = self.loader.s333_score.drop_duplicates(subset=['StudentID'])
        print("✓ ลบแถวซ้ำแล้ว")

        # จัดการค่าว่าง (missing values)
        score_tables = [
            ('SJ111777', self.loader.sj111_score),
            ('SJ220777', self.loader.sj220_score),
            ('S333777', self.loader.s333_score)
        ]

        for name, df in score_tables:
            missing = df.isnull().sum()
            if missing.any():
                print(f"\n{name} - ค่าว่าง:")
                print(missing[missing > 0])
                # เติมค่าว่างด้วย median ของคอลัมน์นั้น
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True)

        print("\n✓ จัดการค่าว่างเรียบร้อย")
        return self


# ==================== 3. FEATURE ENGINEERING ====================
class FeatureEngineer:
    """สร้างฟีเจอร์ใหม่สำหรับโมเดล"""

    def __init__(self, loader):
        self.loader = loader

    def create_features(self):
        """สร้างฟีเจอร์จากข้อมูลดิบ"""
        print("\n" + "=" * 60)
        print("⚙️  กำลังสร้างฟีเจอร์...")
        print("=" * 60)

        # เริ่มจากตาราง ID_Group
        df = self.loader.id_group.copy()

        # ====== ฟีเจอร์จากวิชา SJ111777 ======
        df = self._add_course_features(df, 'SJ111777',
                                       self.loader.sj111_att,
                                       self.loader.sj111_score)

        # ====== ฟีเจอร์จากวิชา SJ220777 ======
        df = self._add_course_features(df, 'SJ220777',
                                       self.loader.sj220_att,
                                       self.loader.sj220_score)

        # ====== ฟีเจอร์จากวิชา SJ333777 ======
        df = self._add_course_features(df, 'SJ333777',
                                       self.loader.sj333_att,
                                       self.loader.s333_score,
                                       has_quiz=False)

        # ====== สร้างฟีเจอร์รวม (Aggregated Features) ======
        att_cols = [c for c in df.columns if c.endswith('_attendance_rate')]
        if att_cols:
            df['avg_attendance_rate'] = df[att_cols].mean(axis=1)
            df['min_attendance_rate'] = df[att_cols].min(axis=1)
            df['max_attendance_rate'] = df[att_cols].max(axis=1)

        score_cols = [c for c in df.columns if '_score' in c and c != 'total_avg_score']
        if score_cols:
            df['total_avg_score'] = df[score_cols].mean(axis=1)
            df['score_std'] = df[score_cols].std(axis=1)
            df['score_min'] = df[score_cols].min(axis=1)
            df['score_max'] = df[score_cols].max(axis=1)

        print(f"\n✓ สร้างฟีเจอร์เสร็จแล้ว: {df.shape[1]} คอลัมน์")
        print(f"✓ จำนวนนักศึกษา: {len(df)}")

        self.feature_df = df
        return self

    def _add_course_features(self, df, course_name, att_df, score_df, has_quiz=True):
        """สร้างฟีเจอร์สำหรับแต่ละวิชา"""
        prefix = course_name.lower()

        # Attendance features
        week_cols = [c for c in att_df.columns if c.startswith('Week')]
        att_features = att_df.copy()
        att_features[f'{prefix}_attendance_rate'] = att_features[week_cols].mean(axis=1)
        att_features[f'{prefix}_attendance_count'] = att_features[week_cols].sum(axis=1)
        att_features[f'{prefix}_late_attendance'] = (att_features[week_cols].iloc[:, :5].mean(axis=1) <
                                                      att_features[week_cols].iloc[:, 5:].mean(axis=1)).astype(int)

        df = df.merge(att_features[['StudentID', f'{prefix}_attendance_rate',
                                     f'{prefix}_attendance_count', f'{prefix}_late_attendance']],
                      on='StudentID', how='left')

        # Score features
        score_features = score_df[['StudentID']].copy()

        # คะแนนแต่ละส่วน (normalized)
        for col in score_df.columns:
            if col != 'StudentID' and col != 'ScoreGrade(100Points)':
                # Extract max points from column name
                max_points = int(col.split('(')[1].split('Points')[0])
                normalized_col = f"{prefix}_{col.split('(')[0].lower()}_norm"
                score_features[normalized_col] = score_df[col] / max_points

        # คะแนนรวม
        if 'ScoreGrade(100Points)' in score_df.columns:
            score_features[f'{prefix}_score'] = score_df['ScoreGrade(100Points)']

        df = df.merge(score_features, on='StudentID', how='left')

        print(f"  - เพิ่มฟีเจอร์จาก {course_name}")
        return df

    def prepare_for_modeling(self, target_course='SJ333777'):
        """เตรียมข้อมูลสำหรับโมเดล - ใช้วิชา SJ333777 เป็น target"""
        print("\n" + "=" * 60)
        print(f"🎯 เตรียมข้อมูลสำหรับโมเดล (Target: {target_course})")
        print("=" * 60)

        df = self.feature_df.copy()
        target_col = f'{target_course.lower()}_score'

        # ลบนักศึกษาที่ไม่มีข้อมูล target
        df = df.dropna(subset=[target_col])
        print(f"✓ นักศึกษาที่มีข้อมูลครบ: {len(df)}")

        # แยก features และ target
        exclude_cols = ['StudentID', target_col] + [c for c in df.columns if 'sj333' in c]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # จัดการค่าว่างที่เหลือ
        X = X.fillna(X.median())

        print(f"\n✓ Features: {X.shape[1]} คอลัมน์")
        print(f"✓ Target: {target_col}")
        print(f"✓ Samples: {len(X)}")

        self.X = X
        self.y = y
        self.feature_cols = feature_cols
        self.target_col = target_col

        return self


# ==================== 4. EXPLORATORY DATA ANALYSIS ====================
class EDA:
    """วิเคราะห์และสร้างกราฟ"""

    def __init__(self, feature_engineer):
        self.fe = feature_engineer
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)

    def analyze(self):
        """วิเคราะห์ข้อมูล"""
        print("\n" + "=" * 60)
        print("📊 กำลังวิเคราะห์ข้อมูล...")
        print("=" * 60)

        self.plot_target_distribution()
        self.plot_correlation_matrix()
        self.plot_feature_importance_preview()
        self.plot_gpa_vs_target()
        self.plot_attendance_vs_target()

        print(f"\n✓ บันทึกกราฟทั้งหมดไว้ที่ {self.output_dir}/")
        return self

    def plot_target_distribution(self):
        """กราฟแสดงการกระจายของคะแนนที่จะทำนาย"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(self.fe.y, bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Score (ScoreGrade)')
        axes[0].set_ylabel('Number of Students')
        axes[0].set_title('Score Distribution for SJ333777')
        axes[0].axvline(self.fe.y.mean(), color='red', linestyle='--', label=f'Mean: {self.fe.y.mean():.2f}')
        axes[0].legend()

        # Box plot
        axes[1].boxplot(self.fe.y, vert=True)
        axes[1].set_ylabel('Score')
        axes[1].set_title('Score Box Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ target_distribution.png")

    def plot_correlation_matrix(self):
        """กราฟ Correlation Matrix"""
        # เลือกฟีเจอร์สำคัญมาแสดง
        important_features = ['GPA', 'avg_attendance_rate', 'total_avg_score',
                             'sj111777_score', 'sj220777_score'] + [self.fe.target_col]
        available_features = [f for f in important_features if f in self.fe.feature_df.columns]

        corr_data = self.fe.feature_df[available_features].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=0.5)
        plt.title('Correlation Matrix - Key Features')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ correlation_matrix.png")

    def plot_feature_importance_preview(self):
        """แสดงความสัมพันธ์ระหว่างฟีเจอร์กับ target"""
        # คำนวณ correlation กับ target
        correlations = []
        for col in self.fe.X.columns:
            corr = np.corrcoef(self.fe.X[col], self.fe.y)[0, 1]
            correlations.append((col, abs(corr)))

        correlations = sorted(correlations, key=lambda x: x[1], reverse=True)[:15]
        features, values = zip(*correlations)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), values, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Absolute Correlation with Target')
        plt.title('Top 15 Features by Correlation')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ feature_correlation.png")

    def plot_gpa_vs_target(self):
        """GPA vs คะแนนที่ทำนาย"""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.fe.feature_df['GPA'], self.fe.y, alpha=0.6, edgecolor='black')
        plt.xlabel('GPA')
        plt.ylabel(f'คะแนน {self.fe.target_col}')
        plt.title('GPA vs Target Score Relationship')

        # Add trend line
        z = np.polyfit(self.fe.feature_df.loc[self.fe.y.index, 'GPA'], self.fe.y, 1)
        p = np.poly1d(z)
        plt.plot(self.fe.feature_df['GPA'], p(self.fe.feature_df['GPA']),
                "r--", alpha=0.8, label='Trend line')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gpa_vs_target.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ gpa_vs_target.png")

    def plot_attendance_vs_target(self):
        """Attendance vs คะแนนที่ทำนาย"""
        if 'avg_attendance_rate' not in self.fe.feature_df.columns:
            return

        plt.figure(figsize=(8, 6))
        plt.scatter(self.fe.feature_df.loc[self.fe.y.index, 'avg_attendance_rate'] * 100,
                   self.fe.y, alpha=0.6, edgecolor='black', color='orange')
        plt.xlabel('อัตราการเข้าเรียนMean (%)')
        plt.ylabel(f'คะแนน {self.fe.target_col}')
        plt.title('Attendance Rate vs Target Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attendance_vs_target.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ attendance_vs_target.png")


# ==================== 5. MODELING ====================
class ModelTrainer:
    """สร้างและเทรนโมเดล"""

    def __init__(self, feature_engineer, test_size=0.2, random_state=42):
        self.fe = feature_engineer
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.output_dir = Path('models')
        self.output_dir.mkdir(exist_ok=True)

    def train_models(self):
        """เทรนหลายโมเดลและเปรียบเทียบ"""
        print("\n" + "=" * 60)
        print("🤖 กำลังเทรนโมเดล...")
        print("=" * 60)

        # แบ่งข้อมูล train/test
        X_train, X_test, y_train, y_test = train_test_split(
            self.fe.X, self.fe.y, test_size=self.test_size, random_state=self.random_state
        )

        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")

        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # รายการโมเดล
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        }

        # เทรนและประเมินแต่ละโมเดล
        print("\nผลการประเมินโมเดล:")
        print("-" * 60)

        for name, model in models.items():
            # Train
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # Evaluate
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5,
                                       scoring='r2')

            self.models[name] = model
            self.results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred_test': y_pred_test
            }

            print(f"\n{name}:")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²:  {test_r2:.4f}")
            print(f"  RMSE:     {test_rmse:.4f}")
            print(f"  MAE:      {test_mae:.4f}")
            print(f"  CV R² :   {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled

        return self

    def select_best_model(self):
        """เลือกโมเดลที่ดีที่สุด"""
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['test_r2'])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]

        print("\n" + "=" * 60)
        print(f"🏆 โมเดลที่ดีที่สุด: {best_name}")
        print("=" * 60)
        print(f"Test R²: {self.results[best_name]['test_r2']:.4f}")
        print(f"RMSE: {self.results[best_name]['test_rmse']:.4f}")

        return self

    def save_best_model(self):
        """บันทึกโมเดลที่ดีที่สุด"""
        model_path = self.output_dir / 'best_model.pkl'
        scaler_path = self.output_dir / 'scaler.pkl'

        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        # บันทึกรายชื่อฟีเจอร์
        feature_path = self.output_dir / 'feature_names.txt'
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.fe.feature_cols))

        print(f"\n✓ บันทึกโมเดลที่ {model_path}")
        print(f"✓ บันทึก scaler ที่ {scaler_path}")
        print(f"✓ บันทึกรายชื่อฟีเจอร์ที่ {feature_path}")

        return self

    def plot_model_comparison(self):
        """เปรียบเทียบผลลัพธ์ของโมเดล"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # R² comparison
        names = list(self.results.keys())
        train_r2 = [self.results[n]['train_r2'] for n in names]
        test_r2 = [self.results[n]['test_r2'] for n in names]

        x = np.arange(len(names))
        width = 0.35

        axes[0, 0].bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Model Comparison: R² Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # RMSE comparison
        rmse = [self.results[n]['test_rmse'] for n in names]
        axes[0, 1].bar(names, rmse, color='coral', alpha=0.8)
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Model Comparison: RMSE')
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)

        # Prediction vs Actual for best model
        best_pred = self.results[self.best_model_name]['y_pred_test']
        axes[1, 0].scatter(self.y_test, best_pred, alpha=0.6, edgecolor='black')
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()],
                       [self.y_test.min(), self.y_test.max()],
                       'r--', lw=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('Actual Score')
        axes[1, 0].set_ylabel('Predicted Score')
        axes[1, 0].set_title(f'Prediction vs Actual ({self.best_model_name})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals
        residuals = self.y_test - best_pred
        axes[1, 1].scatter(best_pred, residuals, alpha=0.6, edgecolor='black', color='green')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Predicted Score')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path('analysis_output') / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ บันทึกกราฟเปรียบเทียบโมเดล")

        return self

    def plot_feature_importance(self):
        """แสดง Feature Importance สำหรับโมเดล Tree-based"""
        if 'Forest' in self.best_model_name or 'Boosting' in self.best_model_name:
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]

            plt.figure(figsize=(10, 6))
            plt.barh(range(15), importances[indices], color='teal')
            plt.yticks(range(15), [self.fe.feature_cols[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Features ({self.best_model_name})')
            plt.tight_layout()
            plt.savefig(Path('analysis_output') / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ บันทึกกราฟ Feature Importance")

        return self


# ==================== 6. MAIN PIPELINE ====================
def main():
    """Run complete pipeline"""
    print("\n")
    print("=" * 60)
    print("  STUDENT GRADE PREDICTION PROJECT")
    print("  ทำนายผลการเรียนของนักศึกษา")
    print("=" * 60)

    # 1. Load data
    loader = DataLoader().load_all()

    # 2. Clean data
    cleaner = DataCleaner(loader).clean()

    # 3. Feature engineering
    fe = FeatureEngineer(loader).create_features().prepare_for_modeling()

    # 4. EDA
    eda = EDA(fe).analyze()

    # 5. Train models
    trainer = ModelTrainer(fe).train_models()
    trainer.select_best_model()
    trainer.save_best_model()
    trainer.plot_model_comparison()
    trainer.plot_feature_importance()

    # 6. Summary
    print("\n" + "=" * 60)
    print("✅ สำเร็จ! Pipeline ทำงานเสร็จสมบูรณ์")
    print("=" * 60)
    print("\n📁 ไฟล์ที่สร้าง:")
    print("  - analysis_output/ : กราฟและการวิเคราะห์")
    print("  - models/ : โมเดลที่เทรนแล้ว")
    print("\n💡 Insights:")
    print("  ✓ นักศึกษาที่มี GPAสูง คะแนนสอบดี เข้าเรียนสม่ำเสมอ")
    print("    มีแนวโน้มที่จะได้เกรดที่ดีในเทอมถัดไป")
    print(f"  ✓ โมเดลสามารถอธิบายความแปรปรวนของคะแนนได้ {trainer.results[trainer.best_model_name]['test_r2']*100:.1f}%")
    print(f"  ✓ ความคลาดเคลื่อนMean (MAE): {trainer.results[trainer.best_model_name]['test_mae']:.2f} คะแนน")
    print("\n🎯 ควรทำอะไรต่อ:")
    print("  1. ใช้โมเดลทำนายล่วงหน้าเพื่อเฝ้าระวังกลุ่มเสี่ยง")
    print("  2. สร้างระบบแจ้งเตือนอาจารย์เมื่อนักศึกษามีแนวโน้มได้เกรดต่ำ")
    print("  3. วิเคราะห์ปัจจัยที่มีผลต่อการเรียนเพื่อปรับปรุงการสอน")
    print("  4. ติดตามและประเมินผลการใช้โมเดลอย่างต่อเนื่อง")
    print("=" * 60)


if __name__ == '__main__':
    main()
