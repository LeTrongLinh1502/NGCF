import torch
from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *
def predict_top_items_for_user(user_id, model, top_k=10):
  
    # Assuming ITEM_NUM is the total number of items in your dataset
    all_item_ids = range(ITEM_NUM)

    # Filter out items that the user has interacted with (in training set)
    try:
        training_items = set(data_generator.train_items[user_id])
        candidate_items = list(filter(lambda item_id: item_id not in training_items, all_item_ids))
    except Exception as e:
        # Handle the case where the user is not in the training set
        candidate_items = all_item_ids

    # Create input tensors for the user and candidate items
    user_tensor = torch.LongTensor([user_id]).to(args.device)
    item_tensor = torch.LongTensor(candidate_items).to(args.device)

    # Predict scores for the candidate items for the given user
    with torch.no_grad():
        user_embedding, item_embeddings, _ = model(user_tensor, item_tensor, None, drop_flag=False)
        scores = torch.matmul(user_embedding, item_embeddings.t())

    # Get the top-k item IDs
    _, top_item_indices = torch.topk(scores.squeeze(), k=top_k)
    top_item_ids = [candidate_items[i] for i in top_item_indices]

    return top_item_ids
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
    model_path = "D:/NGCF-PyTorch-master/model/model_gowalla_final.pth"

    # Tạo mô hình và nạp trọng số
    model.load_state_dict(torch.load(model_path))
    model.eval()

    user_id = 16  # Replace with the actual user ID
    top_k = 20  # Replace with the desired number of top items

    top_item_ids = predict_top_items_for_user(user_id, model, top_k)
    print("Top {} items for user {}: {}".format(top_k, user_id, top_item_ids))

    # user_id_to_predict = 119

    # # Chuyển user_id thành tensor
    # user_tensor = torch.LongTensor([user_id_to_predict]).to(args.device)

    # # Lấy embedding của người dùng
    # user_embedding, _, _ = model(user_tensor, None, None, drop_flag=True)

    # # Lấy embedding của tất cả các mặt hàng
    # item_embeddings = model.item_embedding.weight.to(args.device)
    
    # user_embedding_reshaped = user_embedding.view(1, -1)  # Kích thước: (1, 256)

    # # Tính toán điểm số cho tất cả các mặt hàng
    # scores = torch.matmul(user_embedding_reshaped, item_embeddings.t())

    # # Chuyển scores về numpy array để xử lý
    # scores_np = scores.cpu().detach().numpy()

    # # Lấy indices của 10 mặt hàng có điểm số cao nhất
    # top_10_items = np.argsort(scores_np[0])[::-1][:10]

    # print("Top 10 predicted item IDs:")
    # print(top_10_items)





