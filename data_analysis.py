# alaaraji_math/data_analysis.py

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt


class DataAnalysis:
    def __init__(self, data):
        """إعداد البيانات في إطار بيانات Pandas."""
        self.data = pd.DataFrame(data)

    def summary_statistics(self):
        """توليد إحصائيات وصفية لمجموعة البيانات."""
        return self.data.describe()

    def correlation_matrix(self):
        """حساب مصفوفة الارتباط."""
        return self.data.corr()

    def plot_correlation_matrix(self):
        """رسم مصفوفة الارتباط باستخدام Seaborn."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def perform_t_test(self, column1, column2):
        """إجراء اختبار T بين عمودين."""
        t_statistic, p_value = stats.ttest_ind(self.data[column1].dropna(), self.data[column2].dropna())
        return t_statistic, p_value

    def plot_time_series(self, time_column, value_column):
        """رسم بيانات السلاسل الزمنية."""
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.data, x=time_column, y=value_column)
        plt.title(f'Time Series Plot of {value_column} over {time_column}')
        plt.xlabel(time_column)
        plt.ylabel(value_column)
        plt.show()

    def category_analysis(self, category_column):
        """تحليل بيانات الفئات وعرض تكرار كل فئة."""
        counts = self.data[category_column].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f'Category Analysis for {category_column}')
        plt.xlabel(category_column)
        plt.ylabel('Frequency')
        plt.show()
