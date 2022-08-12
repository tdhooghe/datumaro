import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# %%
def plot_result_bars(data, title, labels, palette=None):
    # , palette="ch:.25"
    g = sns.barplot(data=data, x="class", y="map50", hue="model", palette=palette)
    g.set_xticklabels(labels=g.get_xticklabels(), rotation=45, ha='right')
    handles, _ = g.get_legend_handles_labels()
    g.legend(title='trained on', handles=handles, labels=labels)
    sns.move_legend(g, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    sns.set_style("darkgrid")
    plt.title(title)

    plt.ylabel('mAP@.5')
    plt.ylim(0.45, 1)
    plt.tight_layout()
    plt.show()


path = '/home/thomas/Documents/thesis/results_data_aggregation.csv'

dataframe = pd.read_csv(path)

supervisor_055 = ['coco_relevant', 'guns_coco_055', 'guns_milveh_coco_055']
supervisor_021 = ['coco_relevant', 'guns_coco_021', 'guns_milveh_coco_021']
supervisor_naive = ['naive_merge', 'guns_milveh_coco_055', 'guns_milveh_coco_021']
supervisor_labels = ['coco', 'coco+guns', 'coco+guns+mil_veh']
supervisor_naive_labels = ['naive', 'supervisor@.55conf', 'supervisor@.21conf']

plot_result_bars(dataframe[dataframe.model.isin(supervisor_055)], 'Supervisor model learning progress @.55 conf.',
                 supervisor_labels, palette='Greens')
plot_result_bars(dataframe[dataframe.model.isin(supervisor_021)], 'Supervisor model learning progress @.21 conf.',
                 supervisor_labels, palette='Greens')
plot_result_bars(dataframe[dataframe.model.isin(supervisor_naive)], 'Comparison between supervisor and naive merge',
                 supervisor_naive_labels, palette='gist_earth')
