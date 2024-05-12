# 工作流程
# 1. 先评测了若干方法，形成了 correct 和 wrong list
# 2. 确认经过科学实验后，我们认为的最佳方法会是哪个
# 3. 运行本脚本，从视觉上对比最佳方法的优越性在哪
from pydantic import BaseModel
dataset_name = "nordlands"
class QueryResult(BaseModel):
    query:str
    first_matched_database:str # 只可视化第一个
    correct:bool = True
    
class Method(BaseModel):
    name:str = "method"
    is_our_method:bool = False
    log_dir:str = "log_dir"
    dataset_name:str = dataset_name
    correct_name:str = "corrects.txt"
    wrong_name:str = "wrongs.txt"
    @property
    def correct_path(self):
        return self.log_dir + self.correct_name
    @property
    def wrong_path(self):
        return self.log_dir + self.wrong_name
    def load_data(self)->dict[str, QueryResult]:
        file_name = self.correct_path if self.is_our_method else self.wrong_path
        # result = QueryResult(correct=self.is_our_method) # 对于我们的方法，只筛选正确的
        results = dict()
        with open(file_name, 'r') as fr:
            for line in fr.readlines():
                curLine = line.split(" ")
                result = QueryResult(query=curLine[0], first_matched_database=curLine[1], correct=self.is_our_method)
                results[curLine[0]] = result
        return results
    
other_methods = [
    # Method(name="NetVLAD", log_dir="./NetVLAD/"),
    # Method(name="SFRS", log_dir="./sfrs/"),
    # Method(name="CosPlace", log_dir="./CosPlace/"),
    # Method(name="EigenPlace", log_dir="./eigenplace/"),
    Method(name="Freeze-8", log_dir="./logs/default/2024-05-01_03-39-01"),
] # 会加载 wrongs.txt
our_method = Method(name="Freeze-8 (Ours) ", log_dir="./logs/default/2024-05-01_03-39-01",
                    is_our_method=True) # 会加载 corrects.txt

other_methods_wrong_results:list[dict[str, QueryResult]] = [method.load_data() for method in other_methods]
our_method_correct_results:dict[str, QueryResult] = our_method.load_data()


# class ComparisonTableRow():
#     query:str
#     correct_method_database:str
#     wrong_method_databases:list[str] = []

result = []
for query, query_result in our_method_correct_results.items():
    all_other_wrong = True
    comparison = dict(query=query)
    comparison[our_method.name] = query_result.first_matched_database, 
    for method, method_wrong_results in zip(other_methods, other_methods_wrong_results):
        if query not in method_wrong_results:
            all_other_wrong = False
            break
        else:
            # comparison.wrong_method_databases.append(method_wrong_results[query].first_matched_database)
            comparison[method.name] = method_wrong_results[query].first_matched_database
    if all_other_wrong:
        print(f"Bingo! Critical query {query} is predicted correctly only by our method! ")
        result.append(comparison)

import pandas as pd
df = pd.DataFrame(result)
df.to_csv(f"{dataset_name}_comparison.csv", index=False)

#%%
# from visualizations import build_prediction_image

# prediction_image = build_prediction_image(list_of_images_paths, preds_correct)

#%%
# 根据人看到的图片去看难点。
df['query_instance_challenge_point'] = ['Viewpoint changes', 'Lighting changes']

        
#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

dummy_image_path = '../VPR-datasets-downloader/datasets/pitts30k/images/test/queries/@0584334.39@4476910.88@17@T@040.43857@-080.00562@004561@00@@@@@@pitch1_yaw1@.jpg'
dummy_image_path = 'figures/canton-p1.jpg'
dummy_data = {
    'query_instance_challenge_point': ['Viewpoint changes', 'Lighting changes'],
    'query':[dummy_image_path, dummy_image_path], 
    'NovelMethod (Ours)': [dummy_image_path, dummy_image_path],
    'Compared Method': [dummy_image_path, dummy_image_path]
}

dummy_df = pd.DataFrame(dummy_data)
dummy_df
#%%
dummy_df.to_csv('dummy.csv', index=False)
#%%
def draw_comparisons_df_by_latex(df:pd.DataFrame)->str:
    rows, cols = df.shape
    df = df.copy()
    df.columns = [' ']+list(df.columns[1:])
    for i in range(rows):
        for j in range(cols):
            df_value = df.iloc[i, j]
            latex_item = df_value
            if j == 1:
                latex_item = f'\\smallimage{{{df_value}}}'
            if j == 2:
                latex_item = f'\\greenframe{{{df_value}}}'
            elif j>1:
                latex_item = f'\\redframe{{{df_value}}}'
            df.iloc[i, j] = latex_item
    # cols = cols -1 
    col_content = ' & '.join(df.columns)
    middle_content = '\\\\\n'.join([' & '.join(row) for row in df.values]) 
    return f"""\\noindent
    \\begin{{tabular}}{{{' '.join('c'* cols)}}}
    {col_content}\\\\
    {middle_content}\\\\
    \\end{{tabular}}
    """
    # \\begin{{tabular}}{{|{'|'.join('c'* cols)}|}}
    
    
print(draw_comparisons_df_by_latex(dummy_df))

#%%
# from matplotlib.patches import Rectangle
# # def draw_comparisons_df(df:pd.DataFrame):
# df = dummy_df
# rows, cols = df.shape
# cols = cols -1 
# fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))  # 2x2的子图网格
# for index, axe in np.ndenumerate(axes):
#     # axe.axis('off') 
#     image = Image.open(df.iloc[index[0], index[1]+1])
#     axe.imshow(image)
    
#     bbox = axe.get_position()
#     print(bbox.__dict__)
#     rect = Rectangle((bbox.x0, bbox.y0), bbox.width, bbox.height,
#                  linewidth=2, edgecolor='red', facecolor='none')
#     axe.add_patch(rect)

# 显示图表
# plt.tight_layout()

#%%
# bar_width = 0.25
# 设置x轴的刻度和标签
# fig.set_xticks(df.columns[1:])

# # 设置y轴的标签
# ax.set_ylabel('Performance Score')

# # 添加图例
# ax.legend()

# # 添加网格线
# ax.grid(axis='y', linestyle='--', alpha=0.7)
# fig.subplots_adjust(hspace=0.5, wspace=0.5)

# # 为每一列添加标签
# column_labels = ['Column 1', 'Column 2', 'Column 3']
# for i, label in enumerate(column_labels):
#     # 在网格的最后一行，每个列的中心位置添加标签
#     fig.text(0.5 * i, -0.1, label, ha='center', fontsize=12)

# # 为每一行添加标签
# row_labels = ['Row 1', 'Row 2']
# for j, label in enumerate(row_labels):
#     # 在网格的第一列，每个行的顶部位置添加标签
#     fig.text(-0.1, 0.5 * j, label, va='center', rotation='vertical', fontsize=12)
# fig.tight_layout()
#%%
     

# # 将DataFrame转换为4张图片的列表
# images_ours = df['NovelMethod (Ours)'].tolist()
# images_compared = df['Compared Method'].tolist()

# # 创建画布和子图
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2的子图网格

# # 遍历子图，为每个挑战类型加载和显示图片
# for ax, (challenge, img_ours, img_compared) in zip(axes.flatten(), zip(df.index, images_ours, images_compared)):
#     # 显示我们的方法的图片
#     img = Image.open(img_ours)
#     ax.imshow(img)
#     ax.set_title('Novel Method (Ours)')
#     ax.axis('off')

#     # 在同一个子图中显示比较方法的图片，调整子图布局以容纳两张图片
#     ab = AnnotationBbox(img, (0, 0), frameon=False)
#     ax.add_artist(ab)

#     # 设置挑战点的标题
#     ax.set_xlabel(challenge)

# # 调整子图间距
# plt.tight_layout()
# plt.show()
# %%
