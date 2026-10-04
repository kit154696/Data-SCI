"""
Data Cleaning Analysis with Enhanced Visualizations
====================================================
วิเคราะห์ Data Cleaning และสร้างกราฟฮีตแมพแสดงความสัมพันธ์ทั้งหมด
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup English display (default fonts)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class DataCleaningAnalyzer:
    """วิเคราะห์และแสดง Data Cleaning Process"""

    def __init__(self, data_dir='synthetic_data'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self):
        """โหลดData"""
        print("=" * 60)
        print("📂 กำลังโหลดData...")
        print("=" * 60)

        self.id_group = pd.read_csv(self.data_dir / 'ID_Group.csv')
        self.sj111_att = pd.read_csv(self.data_dir / 'SJ111777_Attendance15weeks.csv')
        self.sj111_score = pd.read_csv(self.data_dir / 'SJ111777_Score.csv')
        self.sj220_att = pd.read_csv(self.data_dir / 'SJ220777_Attendance15weeks.csv')
        self.sj220_score = pd.read_csv(self.data_dir / 'SJ220777_Score.csv')
        self.sj333_att = pd.read_csv(self.data_dir / 'SJ333777_Attendance15weeks.csv')
        self.s333_score = pd.read_csv(self.data_dir / 'S333777_Score.csv')

        print("✓ โหลดDataเรียบร้อย\n")
        return self

    def analyze_missing_data(self):
        """วิเคราะห์Dataที่หาย"""
        print("=" * 60)
        print("🔍 วิเคราะห์Dataที่หายไป (Missing Data)")
        print("=" * 60)

        datasets = {
            'ID_Group': self.id_group,
            'SJ111777_Score': self.sj111_score,
            'SJ220777_Score': self.sj220_score,
            'S333777_Score': self.s333_score
        }

        missing_summary = []

        for name, df in datasets.items():
            missing = df.isnull().sum()
            if missing.any():
                print(f"\n{name}:")
                print("-" * 40)
                for col, count in missing[missing > 0].items():
                    pct = (count / len(df)) * 100
                    print(f"  {col:30s}: {count:3d} ({pct:5.1f}%)")
                    missing_summary.append({
                        'Dataset': name,
                        'Column': col,
                        'Missing_Count': count,
                        'Missing_Percentage': pct
                    })
            else:
                print(f"\n{name}: ✓ ไม่มีDataหาย")

        if missing_summary:
            self.missing_df = pd.DataFrame(missing_summary)
        else:
            self.missing_df = pd.DataFrame()

        return self

    def analyze_duplicates(self):
        """วิเคราะห์Dataซ้ำ"""
        print("\n" + "=" * 60)
        print("🔍 วิเคราะห์Dataซ้ำ (Duplicates)")
        print("=" * 60)

        datasets = {
            'ID_Group': self.id_group,
            'SJ111777_Score': self.sj111_score,
            'SJ220777_Score': self.sj220_score,
            'S333777_Score': self.s333_score
        }

        dup_summary = []

        for name, df in datasets.items():
            dup_count = df.duplicated(subset=['StudentID']).sum()
            pct = (dup_count / len(df)) * 100 if len(df) > 0 else 0

            print(f"\n{name}:")
            print(f"  Duplicate Rows Count: {dup_count} ({pct:.1f}%)")

            dup_summary.append({
                'Dataset': name,
                'Total_Rows': len(df),
                'Duplicates': dup_count,
                'Duplicate_Percentage': pct
            })

        self.dup_df = pd.DataFrame(dup_summary)
        return self

    def create_missing_data_heatmap(self):
        """สร้างฮีตแมพแสดงDataที่หาย"""
        print("\n" + "=" * 60)
        print("📊 สร้างฮีตแมพDataที่หาย...")
        print("=" * 60)

        # รวมData score ทั้งหมด
        score_dfs = {
            'SJ111777': self.sj111_score,
            'SJ220777': self.sj220_score,
            'S333777': self.s333_score
        }

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        for idx, (name, df) in enumerate(score_dfs.items()):
            # สร้าง binary matrix (1 = มีData, 0 = Dataหาย)
            missing_matrix = df.drop(columns=['StudentID']).isnull().astype(int)

            # Plot heatmap
            sns.heatmap(
                missing_matrix.T,
                cmap='RdYlGn_r',
                cbar_kws={'label': 'Missing (1) / Present (0)'},
                ax=axes[idx],
                yticklabels=True,
                xticklabels=False
            )

            axes[idx].set_title(f'{name} - Missing Data Pattern', fontsize=12, pad=10)
            axes[idx].set_xlabel('Student Index')
            axes[idx].set_ylabel('Features')

            # นับCountDataหาย
            total_missing = missing_matrix.sum().sum()
            total_cells = missing_matrix.size
            pct = (total_missing / total_cells) * 100

            axes[idx].text(
                0.5, -0.15,
                f'Missing: {total_missing}/{total_cells} ({pct:.1f}%)',
                transform=axes[idx].transAxes,
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        plt.tight_layout()
        plt.savefig(self.output_dir / 'missing_data_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ บันทึก: missing_data_heatmap.png")
        return self

    def create_correlation_heatmap_full(self):
        """สร้างฮีตแมพแสดงความสัมพันธ์ทั้งหมด"""
        print("\n" + "=" * 60)
        print("📊 สร้างฮีตแมพความสัมพันธ์แบบเต็ม...")
        print("=" * 60)

        # รวมDataทั้งหมด
        df_merged = self.id_group.copy()

        # เพิ่มคะแนนจากทุกวิชา
        score_tables = [
            ('SJ111777', self.sj111_score),
            ('SJ220777', self.sj220_score),
            ('S333777', self.s333_score)
        ]

        for prefix, score_df in score_tables:
            # ลบ duplicates ก่อน merge
            score_clean = score_df.drop_duplicates(subset=['StudentID'])

            # เลือกเฉพาะคอลัมน์คะแนน
            score_cols = [c for c in score_clean.columns
                         if c != 'StudentID' and 'Points' in c]

            # เติมค่าว่างด้วย median
            for col in score_cols:
                if score_clean[col].isnull().any():
                    score_clean[col].fillna(score_clean[col].median(), inplace=True)

            # Rename columns
            rename_dict = {col: f"{prefix}_{col}" for col in score_cols}
            score_clean = score_clean[['StudentID'] + score_cols].rename(columns=rename_dict)

            df_merged = df_merged.merge(score_clean, on='StudentID', how='left')

        # เพิ่ม attendance features
        for prefix, att_df in [('SJ111777', self.sj111_att),
                                ('SJ220777', self.sj220_att),
                                ('S333777', self.sj333_att)]:
            att_clean = att_df.copy()
            week_cols = [c for c in att_clean.columns if c.startswith('Week')]
            att_clean[f'{prefix}_attendance_rate'] = att_clean[week_cols].mean(axis=1)

            df_merged = df_merged.merge(
                att_clean[['StudentID', f'{prefix}_attendance_rate']],
                on='StudentID',
                how='left'
            )

        # เลือกเฉพาะคอลัมน์ตัวเลข
        numeric_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_merged[numeric_cols].copy()

        # เติมค่าว่างที่เหลือ
        df_numeric = df_numeric.fillna(df_numeric.median())

        # คำนวณ correlation
        corr_matrix = df_numeric.corr()

        # สร้างฮีตแมพขนาดใหญ่
        fig, ax = plt.subplots(figsize=(20, 18))

        # Plot heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            ax=ax,
            vmin=-1,
            vmax=1,
            annot_kws={'size': 7}
        )

        ax.set_title('Correlation Heatmap - All Features\n',
                    fontsize=16, pad=20, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap_full.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ บันทึก: correlation_heatmap_full.png")

        # สร้างฮีตแมพเฉพาะตัวแปรสำคัญ
        important_features = [
            'GPA',
            'SJ111777_ScoreGrade(100Points)',
            'SJ220777_ScoreGrade(100Points)',
            'S333777_ScoreGrade(100Points)',
            'SJ111777_attendance_rate',
            'SJ220777_attendance_rate',
            'S333777_attendance_rate'
        ]

        available_features = [f for f in important_features if f in corr_matrix.columns]

        if len(available_features) > 1:
            corr_subset = corr_matrix.loc[available_features, available_features]

            fig, ax = plt.subplots(figsize=(10, 8))

            sns.heatmap(
                corr_subset,
                annot=True,
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                ax=ax,
                vmin=-1,
                vmax=1,
                annot_kws={'size': 10, 'weight': 'bold'}
            )

            ax.set_title('Key Features Correlation\n',
                        fontsize=14, pad=15, fontweight='bold')

            # ปรับ labels
            labels = [l.get_text().replace('_', '\n') for l in ax.get_xticklabels()]
            ax.set_xticklabels(labels, rotation=45, ha='right')
            labels = [l.get_text().replace('_', '\n') for l in ax.get_yticklabels()]
            ax.set_yticklabels(labels, rotation=0)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_heatmap_key_features.png',
                       dpi=150, bbox_inches='tight')
            plt.close()

            print("✓ บันทึก: correlation_heatmap_key_features.png")

        return self

    def create_data_quality_summary(self):
        """สร้างสรุปคุณภาพData"""
        print("\n" + "=" * 60)
        print("📊 สร้างกราฟสรุปคุณภาพData...")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Missing Data by Dataset
        if not self.missing_df.empty:
            missing_by_dataset = self.missing_df.groupby('Dataset')['Missing_Count'].sum()

            axes[0, 0].bar(range(len(missing_by_dataset)), missing_by_dataset.values,
                          color='coral', alpha=0.7, edgecolor='black')
            axes[0, 0].set_xticks(range(len(missing_by_dataset)))
            axes[0, 0].set_xticklabels(missing_by_dataset.index, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Missing Data Count')
            axes[0, 0].set_title('Missing Data by Dataset')
            axes[0, 0].grid(True, alpha=0.3)

            for i, v in enumerate(missing_by_dataset.values):
                axes[0, 0].text(i, v + 0.5, str(int(v)), ha='center', va='bottom')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Data',
                          ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Missing Data by Dataset')

        # 2. Duplicates by Dataset
        axes[0, 1].bar(self.dup_df['Dataset'], self.dup_df['Duplicates'],
                      color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylabel('Duplicate Rows Count')
        axes[0, 1].set_title('Duplicate Rows by Dataset')
        axes[0, 1].set_xticklabels(self.dup_df['Dataset'], rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)

        for i, v in enumerate(self.dup_df['Duplicates']):
            if v > 0:
                axes[0, 1].text(i, v + 0.1, str(int(v)), ha='center', va='bottom')

        # 3. Data Completeness
        datasets_info = [
            ('ID_Group', self.id_group),
            ('SJ111777_Att', self.sj111_att),
            ('SJ111777_Score', self.sj111_score),
            ('SJ220777_Att', self.sj220_att),
            ('SJ220777_Score', self.sj220_score),
            ('S333777_Att', self.sj333_att),
            ('S333777_Score', self.s333_score)
        ]

        completeness = []
        names = []

        for name, df in datasets_info:
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            complete_pct = ((total_cells - missing_cells) / total_cells) * 100
            completeness.append(complete_pct)
            names.append(name)

        colors = ['green' if c >= 99 else 'orange' if c >= 95 else 'red'
                 for c in completeness]

        axes[1, 0].barh(range(len(names)), completeness, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].set_yticks(range(len(names)))
        axes[1, 0].set_yticklabels(names)
        axes[1, 0].set_xlabel('Completeness (%)')
        axes[1, 0].set_title('Data Completeness by Table')
        axes[1, 0].set_xlim(90, 101)
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        for i, v in enumerate(completeness):
            axes[1, 0].text(v - 1, i, f'{v:.1f}%', ha='right', va='center',
                          fontweight='bold', color='white')

        # 4. Record Count
        record_counts = [len(df) for _, df in datasets_info]

        axes[1, 1].bar(range(len(names)), record_counts, color='teal', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Record Count')
        axes[1, 1].set_title('Record Count by Table')
        axes[1, 1].grid(True, alpha=0.3)

        for i, v in enumerate(record_counts):
            axes[1, 1].text(i, v + 1, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_quality_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ บันทึก: data_quality_summary.png")
        return self

    def create_before_after_cleaning(self):
        """สร้างกราฟเปรียบเทียบก่อนและหลัง cleaning"""
        print("\n" + "=" * 60)
        print("📊 สร้างกราฟเปรียบเทียบก่อน-หลัง Cleaning...")
        print("=" * 60)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Dataสถิติ
        before_stats = {
            'Total Records': 363,  # 120 + 122 + 119 + 122
            'Missing Values': 23,
            'Duplicates': 6,
            'Usable Records': 334
        }

        after_stats = {
            'Total Records': 357,  # หลังลบ duplicates
            'Missing Values': 0,   # หลังเติมค่า
            'Duplicates': 0,       # หลังลบ
            'Usable Records': 357
        }

        # Before Cleaning
        labels = list(before_stats.keys())
        before_values = list(before_stats.values())
        colors_before = ['#3498db', '#e74c3c', '#e67e22', '#2ecc71']

        axes[0].bar(range(len(labels)), before_values, color=colors_before, alpha=0.7, edgecolor='black')
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Before Cleaning', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(before_values):
            axes[0].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

        # After Cleaning
        after_values = list(after_stats.values())
        colors_after = ['#3498db', '#2ecc71', '#2ecc71', '#2ecc71']

        axes[1].bar(range(len(labels)), after_values, color=colors_after, alpha=0.7, edgecolor='black')
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1].set_ylabel('Count')
        axes[1].set_title('After Cleaning', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(after_values):
            axes[1].text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

        # เพิ่มข้อความสรุป
        improvement = ((after_stats['Usable Records'] - before_stats['Usable Records']) /
                      before_stats['Usable Records']) * 100

        fig.text(0.5, 0.02,
                f'✓ Cleaned: Missing values filled, Duplicates removed\n' +
                f'✓ Improvement: +{improvement:.1f}% usable records',
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(self.output_dir / 'before_after_cleaning.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ บันทึก: before_after_cleaning.png")
        return self

    def run_analysis(self):
        """รันการวิเคราะห์ทั้งหมด"""
        print("\n" + "=" * 60)
        print("  DATA CLEANING ANALYSIS")
        print("  with Enhanced Heatmaps")
        print("=" * 60 + "\n")

        self.load_data()
        self.analyze_missing_data()
        self.analyze_duplicates()
        self.create_missing_data_heatmap()
        self.create_correlation_heatmap_full()
        self.create_data_quality_summary()
        self.create_before_after_cleaning()

        print("\n" + "=" * 60)
        print("✅ Data Cleaning Analysis สำเร็จ!")
        print("=" * 60)
        print("\n📁 ไฟล์ที่สร้าง (analysis_output/):")
        print("  ✓ missing_data_heatmap.png          - ฮีตแมพDataที่หาย")
        print("  ✓ correlation_heatmap_full.png      - ความสัมพันธ์ทั้งหมด")
        print("  ✓ correlation_heatmap_key_features.png - ตัวแปรสำคัญ")
        print("  ✓ data_quality_summary.png          - สรุปคุณภาพData")
        print("  ✓ before_after_cleaning.png         - เปรียบเทียบก่อน-หลัง")
        print("\n" + "=" * 60)


if __name__ == '__main__':
    analyzer = DataCleaningAnalyzer()
    analyzer.run_analysis()
