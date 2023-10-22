import torch
from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *
import multiprocessing
import heapq

cores = multiprocessing.cpu_count() // 2



def predict_for_user_batch(user_ids, model, K, drop_flag):
    user_items = []
    for user_id in user_ids:
        try:
            training_items = set(data_generator.train_items[user_id])
            candidate_items = list(filter(lambda item_id: item_id not in training_items, range(ITEM_NUM)))
        except Exception:
            candidate_items = range(ITEM_NUM)

        user_items.append((user_id, candidate_items))

    top_items = []
    with torch.no_grad():
        for user_id, candidate_items in user_items:
            user_tensor = torch.LongTensor([user_id]).to(args.device)
            item_tensor = torch.LongTensor(candidate_items).to(args.device)

            user_embedding, item_embeddings, _ = model(user_tensor, item_tensor, None, drop_flag=drop_flag)
            scores = torch.matmul(user_embedding, item_embeddings.t())

            _, top_item_indices = torch.topk(scores.squeeze(), k=min(K, len(candidate_items)))
            top_items.append((user_id, [candidate_items[i] for i in top_item_indices]))

    return top_items

def predict(model, users_to_predict, K, drop_flag=False, batch_test_flag=False):
    predictions = {}
    pool = multiprocessing.Pool(cores)
    u_batch_size = BATCH_SIZE * 2
    n_test_users = len(users_to_predict)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = min((u_batch_id + 1) * u_batch_size, n_test_users)

        user_batch = users_to_predict[start: end]

        top_items_batch = predict_for_user_batch(user_batch, model, K, drop_flag)

        count += len(top_items_batch)

        for user_id, top_items in top_items_batch:
            predictions[user_id] = top_items

    assert count == n_test_users
    return predictions
if __name__ == '__main__':
    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)
    # Đường dẫn đến mô hình đã lưu
    model_path = "D:/NGCF-PyTorch-master/model_amazon/model_amazon_book.pth"

    # Tạo mô hình và nạp trọng số
    model.load_state_dict(torch.load(model_path))
    model.eval()

    user_id_to_predict = 39

    # Filter the user to predict
    users_to_predict = [user_id_to_predict]

    # Set the desired value of K (number of top items to predict)
    K = 10  # Adjust this value based on your requirement

    # Call the predict function
    predictions = predict(model, users_to_predict, K)

    # Get the top K predicted items for the user
    top_items = predictions[user_id_to_predict]

    # Display the top predicted items for the user
    print(f"Top {K} predicted items for user {user_id_to_predict}: {top_items}")
    





