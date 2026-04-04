# save as: save_plots.py in your project root
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator("../runs/bert4rec").Reload()

for tag in ea.Tags()["scalars"]:
    df = pd.DataFrame(ea.Scalars(tag))
    plt.figure(figsize=(8, 4))
    plt.plot(df["step"], df["value"])
    plt.title(tag)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.tight_layout()
    fname = tag.replace("/", "_") + ".png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"saved {fname}")
