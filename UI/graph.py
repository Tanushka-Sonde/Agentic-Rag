import matplotlib.pyplot as plt

# Data
ratios = ['Return on Equity', 'Return on Capital Employed']
deviations = [5, 1]

# EY brand colors
colors = ['#FFE600', '#2E2E38']

# Create the bar chart
fig, ax = plt.subplots()
ax.bar(ratios, deviations, color=colors)

# Set the title and labels
ax.set_title('Number of Companies Deviating from Standard Formula')
ax.set_xlabel('Ratio')
ax.set_ylabel('Number of Companies')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()