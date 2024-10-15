import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

pal = sns.color_palette("Reds", 5)

intervals = [
    "≤ 11.42",
    "(11.42, 20.04]",
    "(20.04, 28.66]",
    "(28.66, 37.28]",
    "> 37.28",
]

levels = ["Low", "Medium-low", "Medium", "Medium-high", "High"]

fig, ax = plt.subplots()
ax.legend(
    handles=[
        *[mpatches.Patch(color=color, label=levels[i]) for i, color in enumerate(pal)],
        *[mpatches.Patch(color="white", label=interval) for interval in intervals],
    ],
    ncols=2,
    title="Total antibiotic consumption (DDD/1,000/day)",
)

plt.show()
