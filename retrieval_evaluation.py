#%%

#%%
queries_descriptors = all_descriptors[test_ds.database_num:]
database_descriptors = all_descriptors[:test_ds.database_num]

torch.save(queries_descriptors,f"{output_folder}/queries_descriptors.pth")
torch.save(database_descriptors,f"{output_folder}/database_descriptors.pth")


    

# Use a kNN to find predictions
faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
faiss_index.add(database_descriptors)
del database_descriptors, all_descriptors

logging.debug("Calculating recalls")
_, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))

torch.save(predictions,f"{output_folder}/predictions.pth")

# For each query, check if the predictions are correct
positives_per_query = test_ds.get_positives()
recalls = np.zeros(len(args.recall_values))
corrects = [] # [(query_index, database_index), ...]
wrongs = []
for query_index, preds in enumerate(predictions):
    # query_index 是被测试的query图片的索引
    # preds 是这一张图片我们预测出来的最相似的图片的索引
    for i, n in enumerate(args.recall_values):
        exists_correct_in_n = np.any(np.in1d(preds[:n], positives_per_query[query_index]))
        if i == 0:
            if exists_correct_in_n:
                corrects.append((query_index, preds[0]))
            else:
                wrongs.append((query_index, preds[0]))
        if exists_correct_in_n:
            recalls[i:] += 1
            break
with open(f"{output_folder}/corrects.txt", 'w') as f:
    f.writelines([test_ds.queries_paths[query_index]+" "+test_ds.database_paths[correct_pred_index]+"\n"
                  for query_index, correct_pred_index in corrects])
with open(f"{output_folder}/wrongs.txt", 'w') as f:
    f.writelines([test_ds.queries_paths[query_index]+" "+test_ds.database_paths[wrong_pred_index]+"\n"
                  for query_index, wrong_pred_index in wrongs])


# Divide by queries_num and multiply by 100, so the recalls are in percentages
recalls = recalls / test_ds.queries_num * 100
recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
logging.info(recalls_str)

# 导出recalls等指标到json文件
recalls_dict = {f"R@{val}": rec for val, rec in zip(args.recall_values, recalls)}
with open(f"{output_folder}/results.json", 'w') as f:
    json.dump(recalls_dict, f, indent=4)

# Save visualizations of predictions
if args.num_preds_to_save != 0:
    logging.info("Saving final predictions")
    # For each query save num_preds_to_save predictions
    visualizations.save_preds(predictions[:, :args.num_preds_to_save], test_ds,
                              output_folder, args.save_only_wrong_preds)