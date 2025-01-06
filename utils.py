"""
Test Utilities Module

This module provides utilities for testing the classifier, including
report generation and test data handling.
"""

import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


class TestReportGenerator:
    """Generates HTML reports for classification test results"""

    def __init__(self):
        """Initialize the report generator with template configuration"""
        template_dir = Path(__file__).parent / "auto_post_classifier" / "templates"
        print(template_dir)
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12
        )
        
    def generate_report(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Generate an HTML report of the classification test results.
        
        Args:
            df: DataFrame containing classification results and volunteer validations
            output_path: Path to save the generated HTML report
        """
        template = self.env.get_template('report_template.html')
        
        # Prepare data for the template
        results = []
        for _, row in df.iterrows():
            results.append({
                'uuid': row.name,
                'text': row['text'],
                'model_category': row['category'],
                'volunteer_category': row['volunteers'],
                'reason': row['reason'],
                'platform': row['platform'],
                'is_match': row['category'] == row['volunteers']
            })
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate and save report
        html_content = template.render(results=results)
        with open(output_path, 'w') as f:
            f.write(html_content) 

    def generate_pdf_report(self, df: pd.DataFrame, output_path: str = "test_reports/classification_report.pdf"):
        """Generate a comprehensive PDF report with all test results and visualizations"""
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Container for PDF elements
        elements = []
        
        # Title
        elements.append(Paragraph("Classification Test Report", self.title_style))
        elements.append(Spacer(1, 12))
        
        # 1. Confusion Matrix
        elements.append(Paragraph("Confusion Matrix", self.heading_style))
        cm_path = self._generate_confusion_matrix(df)
        elements.append(Image(cm_path, width=400, height=300))
        elements.append(Spacer(1, 20))
        
        # 2. Label Distribution
        elements.append(Paragraph("Label Distributions", self.heading_style))
        dist_path = self._generate_label_distribution(df)
        elements.append(Image(dist_path, width=400, height=200))
        elements.append(Spacer(1, 20))
        
        # 3. Platform Performance
        elements.append(Paragraph("Platform Performance", self.heading_style))
        platform_path = self._generate_platform_performance(df)
        elements.append(Image(platform_path, width=400, height=200))
        elements.append(Spacer(1, 20))
        
        # 4. Classification Results Table
        elements.append(Paragraph("Detailed Results", self.heading_style))
        table_data = self._generate_results_table(df)
        elements.append(table_data)
        
        # Build PDF
        doc.build(elements)
        
        # Cleanup temporary image files
        for path in [cm_path, dist_path, platform_path]:
            if os.path.exists(path):
                os.remove(path)

    def _generate_confusion_matrix(self, df):
        plt.figure(figsize=(10, 8))
        actual_labels = df['volunteers'].tolist()
        predicted_labels = df['category'].tolist()
        unique_labels = sorted(list(set(actual_labels + predicted_labels)))
        
        cm = confusion_matrix(actual_labels, predicted_labels, labels=unique_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_labels,
                    yticklabels=unique_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        path = "temp_confusion_matrix.png"
        plt.savefig(path)
        plt.close()
        return path

    def _generate_label_distribution(self, df):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        df['volunteers'].value_counts().plot(kind='bar')
        plt.title('Distribution of Actual Labels')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        df['category'].value_counts().plot(kind='bar')
        plt.title('Distribution of Predicted Labels')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        path = "temp_distribution.png"
        plt.savefig(path)
        plt.close()
        return path

    def _generate_platform_performance(self, df):
        platforms = df['platform'].unique()
        platform_accuracies = {}
        
        for platform in platforms:
            platform_df = df[df['platform'] == platform]
            accuracy = np.mean(platform_df['volunteers'] == platform_df['category'])
            platform_accuracies[platform] = accuracy
        
        plt.figure(figsize=(10, 6))
        plt.bar(platform_accuracies.keys(), platform_accuracies.values())
        plt.title('Accuracy by Platform')
        plt.xlabel('Platform')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        path = "temp_platform_performance.png"
        plt.savefig(path)
        plt.close()
        return path

    def _generate_results_table(self, df):
        # Create table data
        data = [['Text', 'Platform', 'Model Category', 'Volunteer Category', 'Match']]
        for _, row in df.iterrows():
            data.append([
                row['text'][:100] + '...' if len(row['text']) > 100 else row['text'],
                row['platform'],
                row['category'],
                row['volunteers'],
                '✓' if row['category'] == row['volunteers'] else '✗'
            ])
        
        # Create table with style
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        return table 