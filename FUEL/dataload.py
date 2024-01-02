import os
import random
import numpy as np 
from utils import read_data
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import pdb
import json

class Federated_Dataset(Dataset):
    def __init__(self, X, Y, A):
        self.X = X
        self.Y = Y
        self.A = A

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        A = self.A[index]
        return X, Y, A 

    def __len__(self):
        return self.X.shape[0]


#### adult dataset x("51 White", "52 Asian-Pac-Islander", "53 Amer-Indian-Eskimo", "54 Other", "55 Black", "56 Female", "57 Male")
def LoadDataset(args):
    clients_name, groups, train_data, test_data = read_data(args.train_dir, args.test_dir)

    # client_name [phd, non-phd]
    client_train_loads = []
    client_test_loads = []
    args.n_clients = len(clients_name)
    # clients_name = clients_name[:1]
    if args.dataset == "adult":
        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.sensitive_attr == "race":
                A = X[:,51] # [1: white, 0: other]
                X = np.delete(X, [51, 52, 53, 54, 55], axis = 1)
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "sex":
                A = X[:, 56] # [1: female, 0: male]
                X = np.delete(X, [56, 57], axis = 1)
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "none-race":
                A = X[:, 51]  # [1: white, 0: other]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "none-sex":
                A = X[:, 56]
                args.n_feats = X.shape[1]
            else:
                print("error sensitive attr")
                exit()
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last))

        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            if args.sensitive_attr =="race":
                A = X[:,51] # [1: white, 0: other]
                X = np.delete(X, [51, 52, 53, 54, 55],axis = 1)
            elif args.sensitive_attr == "sex":
                A = X[:, 56] # [1: female, 0: male]
                X = np.delete(X, [56, 57], axis = 1)
            elif args.sensitive_attr == "none-race":
                A = X[:, 51]  # [1: white, 0: other]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "none-sex":
                A = X[:, 56]
                args.n_feats = X.shape[1]
            else:
                print("error sensitive attr")
                exit()

            dataset = Federated_Dataset(X, Y, A)

            client_test_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last)) 

    elif "eicu" in args.dataset:
    # elif args.dataset == "eicu_d" or args.dataset == "eicu_los":
        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.sensitive_attr == "race":
                A = train_data[client]["race"]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "sex":
                A = train_data[client]["gender"]
                args.n_feats = X.shape[1]
            else:
                A = train_data[client]["race"]
                args.n_feats = X.shape[1]
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last))

        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            if args.sensitive_attr =="race":
                A = test_data[client]["race"]
            elif args.sensitive_attr == "sex":
                A = test_data[client]["gender"]
            else:
                A = test_data[client]["race"]

            dataset = Federated_Dataset(X, Y, A)

            client_test_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last)) 

    elif args.dataset == "health":
        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.sensitive_attr == "race":
                A = train_data[client]["race"]
                args.n_feats = X.shape[1]
            elif args.sensitive_attr == "sex":
                A = train_data[client]["isfemale"]
                args.n_feats = X.shape[1]
            else:
                A = train_data[client]["isfemale"]
                args.n_feats = X.shape[1]
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
                                                 shuffle=args.shuffle,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 drop_last=args.drop_last))

        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            if args.sensitive_attr == "race":
                A = test_data[client]["race"]
            elif args.sensitive_attr == "sex":
                A = test_data[client]["isfemale"]
            else:
                A = np.zeros(X.shape[0])

            dataset = Federated_Dataset(X, Y, A)

            client_test_loads.append(DataLoader(dataset, X.shape[0],
                                                shuffle=args.shuffle,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                drop_last=args.drop_last))
    elif args.dataset == "bank":
        for client in clients_name:
            def handle(data, client_loads):
                X = np.array(data[client]["x"]).astype(np.float32)
                Y = np.array(data[client]["y"]).astype(np.float32)
                if args.sensitive_attr == "married":
                    sensitive = 19
                    sensitives = [18, 19, 20]
                elif args.sensitive_attr == "noloan":
                    sensitive = 29
                    sensitives = [29, 30]
                A = X[:, sensitive]
                X = np.delete(X, sensitives, axis = 1)
                args.n_feats = X.shape[1]
                dataset = Federated_Dataset(X, Y, A)
                client_loads.append(DataLoader(dataset, X.shape[0],
                shuffle = args.shuffle,
                num_workers = args.num_workers,
                pin_memory = True,
                drop_last = args.drop_last))
            handle(train_data, client_train_loads)
            handle(test_data, client_test_loads)
    elif args.dataset == "compas":
        for client in clients_name:
            def handle(data, client_loads):
                X = np.array(data[client]["x"]).astype(np.float32)
                Y = np.array(data[client]["y"]).astype(np.float32)
                if args.sensitive_attr == "sex":
                    sensitive = 5
                    sensitives = [5, 6]
                elif args.sensitive_attr == "race":
                    sensitive = 12
                    sensitives = [12, 13, 14, 15, 16, 17]
                A = X[:, sensitive]
                X = np.delete(X, sensitives, axis = 1)
                args.n_feats = X.shape[1]
                dataset = Federated_Dataset(X, Y, A)
                client_loads.append(DataLoader(dataset, X.shape[0],
                shuffle = args.shuffle,
                num_workers = args.num_workers,
                pin_memory = True,
                drop_last = args.drop_last))
            handle(train_data, client_train_loads)
            handle(test_data, client_test_loads)

    if args.mix_dataset:
        print("mix_dataset")
        combined_loaders = []
        for loader1, loader2 in zip(client_train_loads, client_test_loads):
            combined_dataset = ConcatDataset([loader1.dataset, loader2.dataset])
            combined_loader = DataLoader(combined_dataset, 
                                         batch_size=len(combined_dataset), 
                                         shuffle = args.shuffle,
                                         num_workers = args.num_workers,
                                         pin_memory = True,
                                         drop_last = args.drop_last)
            combined_loaders.append(combined_loader)
        return client_train_loads, combined_loaders
    # 返回三个互不相交的数据集，用于作为训练集、测试集、模拟整体的数据集
    if args.new_trial:
        client_train_indices = {}
        if args.valid:
            with open(os.path.join(args.target_dir_name, 'results', 'client_indices.json'), 'r') as f:
                client_train_indices = json.load(f)
        def cut_dataset(dataset, client_name):
            if args.new_trial_train_rate <= 0 or args.new_trial_test_rate <= 0 or args.new_trial_whole_rate <= 0 \
                or args.new_trial_train_rate + args.new_trial_test_rate + args.new_trial_whole_rate > 1.0:
                raise RuntimeError("new trial rate illegal")
            while True:
                class SampleError(Exception):
                    def __init__(self):
                        super().__init__()
                def judge_contain_all_sensitive(sub_dataset):
                    a_con = sum(1 for _, _, a in sub_dataset if a == 1.0)
                    na_con = sum(1 for _, _, a in sub_dataset if a == 0.0)
                    if a_con + na_con != len(sub_dataset):
                        raise "a_con + na_con != len(sub_dataset)"
                    if a_con != 0 and na_con != 0:
                        return True
                    return False
                def sample_in_remain(indices_remain, rate, record_client=None):
                    if record_client != None and args.valid:
                        indices = client_train_indices[record_client]
                        # print(f'{indices[0]} {indices[1]} {indices[2]}')
                    else: 
                        indices = np.random.choice(indices_remain, int(len(dataset) * rate), replace = False)
                    if record_client != None and not args.valid:
                        client_train_indices[record_client] = indices.tolist()
                        # print(f'{indices[0]} {indices[1]} {indices[2]}')
                    sub_dataset = Subset(dataset, indices)
                    if not judge_contain_all_sensitive(sub_dataset):
                        raise SampleError()
                    sub_dataloader = DataLoader(sub_dataset,
                                    batch_size=len(sub_dataset), 
                                    shuffle = args.shuffle,
                                    num_workers = args.num_workers,
                                    pin_memory = True,
                                    drop_last = args.drop_last)
                    indices_remain = list(set(indices_remain) - set(indices))
                    return indices, indices_remain, sub_dataloader
                indices_remain = range(len(dataset))
                try:
                    indices_train, indices_remain, sub_dataloader_train = sample_in_remain(indices_remain, args.new_trial_train_rate, client_name)
                    indices_test, indices_remain, sub_dataloader_test = sample_in_remain(indices_remain, args.new_trial_test_rate)
                    indices_whole, _, sub_dataloader_whole = sample_in_remain(indices_remain, args.new_trial_whole_rate)
                    if set(indices_train) & set(indices_test) or set(indices_test) & set(indices_whole) or set(indices_train) & set(indices_whole):
                        raise RuntimeError('3 dataset is intersect')
                    break
                except SampleError as e:
                    print("dataset only containes one sensitive feature, retry sampling")
                    continue
            return sub_dataloader_train, sub_dataloader_test, sub_dataloader_whole
        new_client_train_loads = []
        new_client_test_loads = []
        new_client_whole_loads = []
        for client_i, (client_train_load, client_test_load) in enumerate(zip(client_train_loads, client_test_loads)):
            combined_dataset = ConcatDataset([client_train_load.dataset, client_test_load.dataset])
            new_client_train_load, new_client_test_load, new_client_whole_load = cut_dataset(combined_dataset, str(client_i))
            # print(f'new_client_train_load rate: {len(new_client_train_load.dataset) * 1.0 / len(combined_dataset)}')
            # print(f'new_client_test_load rate: {len(new_client_test_load.dataset) * 1.0 / len(combined_dataset)}')
            # print(f'new_client_whole_load rate: {len(new_client_whole_load.dataset) * 1.0 / len(combined_dataset)}')
            new_client_train_loads.append(new_client_train_load)
            new_client_test_loads.append(new_client_test_load)
            new_client_whole_loads.append(new_client_whole_load)
        if not args.valid:
            with open(os.path.join(args.target_dir_name, 'results', 'client_indices.json'), 'w') as f:
                json.dump(client_train_indices, f)
        return new_client_train_loads, new_client_test_loads, new_client_whole_loads

    return client_train_loads, client_test_loads

