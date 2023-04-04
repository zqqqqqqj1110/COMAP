import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('2_2.xls')
sns.boxenplot(x='Location', y='Price', data=df) # x与y的改
sns.despine(trim=False, left=True);plt.grid(True)

plt.show()