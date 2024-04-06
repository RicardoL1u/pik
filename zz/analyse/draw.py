import os
import pandas as pd
import matplotlib.pyplot as plt

BBH_ACL_FOLDER = "../../data/bbh/bbh-acl"

BBH_NEW_FOLDER = "../../data/bbh/bbh-new"

# BBH_ACL_FOLDER = "../../../evaluate/data/bbh/bbh-acl"

# BBH_NEW_FOLDER = "../../../evaluate/data/bbh/bbh-new"


def get_file_name(folder):
    name_list = []
    for file in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, file)):
            name_list.append(file)
    print(name_list)


# get_file_name(BBH_NEW_FOLDER)

NOW_FOLDER = BBH_ACL_FOLDER

origin_gemma_file = "pretrained-google-gemma-2b_0.0"

before_dpo_file_list = [
    "output-gemma-2b-v0-20240405-043742-before-dpo-checkpoint-20_0.0",
    "output-gemma-2b-v0-20240405-043742-before-dpo-checkpoint-60_0.0",
    "output-gemma-2b-v0-20240405-043742-before-dpo-checkpoint-100_0.0",
    "output-gemma-2b-v0-20240405-043742-before-dpo-checkpoint-140_0.0",
    "output-gemma-2b-v0-20240405-043742-before-dpo-checkpoint-180_0.0",
    "output-gemma-2b-v0-20240405-043742-before-dpo-checkpoint-220_0.0",
    "output-gemma-2b-v0-20240405-043742-before-dpo-checkpoint-260_0.0",
    "output-gemma-2b-v0-20240405-043742-before-dpo-checkpoint-288_0.0",
]

after_dpo_file_list = [
    "output-gemma-2b-v1-20240405-053307-after-dpo-checkpoint-20_0.0",
    "output-gemma-2b-v1-20240405-053307-after-dpo-checkpoint-60_0.0",
    "output-gemma-2b-v1-20240405-053307-after-dpo-checkpoint-100_0.0",
    "output-gemma-2b-v1-20240405-053307-after-dpo-checkpoint-140_0.0",
    "output-gemma-2b-v1-20240405-053307-after-dpo-checkpoint-180_0.0",
    "output-gemma-2b-v1-20240405-053307-after-dpo-checkpoint-220_0.0",
    "output-gemma-2b-v1-20240405-053307-after-dpo-checkpoint-260_0.0",
    "output-gemma-2b-v1-20240405-053307-after-dpo-checkpoint-288_0.0",
]

fig_length = len(after_dpo_file_list) // 3
if fig_length * 3 < len(after_dpo_file_list):
    fig_length += 1

fig, ax = plt.subplots(fig_length, 3, figsize=(fig_length * 10, 8 * 3), sharey=True)
ax = ax.flatten()

final_df = pd.read_csv(
    os.path.join(NOW_FOLDER, origin_gemma_file, "accuracy.csv"), index_col=0
)
final_df = final_df[["accuracy"]]
final_df.rename(columns={"accuracy": "pretrained"}, inplace=True)

for file in before_dpo_file_list:
    df = pd.read_csv(os.path.join(NOW_FOLDER, file, "accuracy.csv"), index_col=0)
    df = df[["accuracy"]]
    df.rename(
        columns={"accuracy": file.split("-")[-4] + "-" + file.split("-")[-1]},
        inplace=True,
    )
    final_df = pd.concat([final_df, df], axis=1)

for file in after_dpo_file_list:
    df = pd.read_csv(os.path.join(NOW_FOLDER, file, "accuracy.csv"), index_col=0)
    df = df[["accuracy"]]
    df.rename(
        columns={"accuracy": file.split("-")[-4] + "-" + file.split("-")[-1]},
        inplace=True,
    )
    final_df = pd.concat([final_df, df], axis=1)

color_list = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "pink",
    "brown",
    "cyan",
    "magenta",
    "gray",
    "olive",
    "lime",
    "teal",
    "navy",
    "maroon",
    "aqua",
    "fuchsia",
    "silver",
]
color_index = 0
color_map = {}
for column in final_df.columns:
    if "before" in column:
        now_ckpt = int(column.split("-")[1].split("_")[0])
        if now_ckpt not in color_map:
            color_map[now_ckpt] = color_list[color_index]
            color_index += 1
        ax[color_list.index(color_map[now_ckpt])].plot(
            final_df[column],
            label=column,
            marker=".",
            color=color_map[now_ckpt],
            alpha=0.3,
        )
    elif "after" in column:
        now_ckpt = int(column.split("-")[1].split("_")[0])
        if now_ckpt not in color_map:
            color_map[now_ckpt] = color_list[color_index]
            color_index += 1
        ax[color_list.index(color_map[now_ckpt])].plot(
            final_df[column], label=column, marker=".", color=color_map[now_ckpt]
        )


for i in range(color_index):
    ax[i].plot(final_df["pretrained"], label="pretrained", marker=".", color="black")
    ax[i].legend()
    ax[i].set_xticks(final_df.index)
    ax[i].set_xticklabels(
        labels=final_df.index,
        rotation=45,
        rotation_mode="anchor",
        horizontalalignment="right",
    )

fig.subplots_adjust(bottom=0.25)
fig.suptitle(NOW_FOLDER.split("/")[-1])
fig.tight_layout()
fig.savefig(NOW_FOLDER.split("/")[-1] + ".png")

print(final_df.mean(axis=0))
