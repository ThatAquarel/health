import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

pal = sns.color_palette("Reds", 5)

fig, ax = plt.subplots()
ax.legend(handles=[mpatches.Patch(color=color, label="The red data") for color in pal])

plt.show()
