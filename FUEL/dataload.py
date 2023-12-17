import os
import random
import numpy as np 
from utils import read_data
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import pdb

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
def LoadDataset(args, another_half = False):
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
    if args.new_trial:
        def cut_loader(dataloader):
            dataset = dataloader.dataset
            while True:
                def judge_contain_all_sensitive(sub_dataset):
                    a_con = sum(1 for _, _, a in sub_dataset if a == 1.0)
                    na_con = sum(1 for _, _, a in sub_dataset if a == 0.0)
                    if a_con + na_con != len(sub_dataset):
                        raise "a_con + na_con != len(sub_dataset)"
                    if a_con != 0 and na_con != 0:
                        return True
                    return False
                indices_1 = np.random.choice(range(len(dataset)), int(len(dataset) * 0.5), replace = False)
                sub_dataset_1 = Subset(dataset, indices_1)
                if not judge_contain_all_sensitive(sub_dataset_1):
                    print("dataset only containes one sensitive feature, retry sampling")
                    continue
                indices_2 = list(set(range(len(dataset))) - set(indices_1))
                sub_dataset_2 = Subset(dataset, indices_2)
                if not judge_contain_all_sensitive(sub_dataset_2):
                    print("dataset only containes one sensitive feature, retry sampling")
                    continue
                break
            sub_dataloader_1 = DataLoader(sub_dataset_1,
                                    batch_size=len(sub_dataset_1), 
                                    shuffle = args.shuffle,
                                    num_workers = args.num_workers,
                                    pin_memory = True,
                                    drop_last = args.drop_last)
            sub_dataloader_2 = DataLoader(sub_dataset_2,
                                    batch_size=len(sub_dataset_2), 
                                    shuffle = args.shuffle,
                                    num_workers = args.num_workers,
                                    pin_memory = True,
                                    drop_last = args.drop_last)
            return sub_dataloader_1, sub_dataloader_2
        if not another_half:
            # def cut_loader(dataloader):
            #     dataset = dataloader.dataset
            #     while True:
            #         indices = np.random.choice(range(len(dataset)), int(len(dataset) * 0.5), replace = False)
            #         sub_dataset = Subset(dataset, indices)
            #         a_con = sum(1 for _, _, a in sub_dataset if a == 1.0)
            #         na_con = sum(1 for _, _, a in sub_dataset if a == 0.0)
            #         if a_con + na_con != len(sub_dataset):
            #             raise "a_con + na_con != len(sub_dataset)"
            #         if a_con != 0 and na_con != 0:
            #             break
            #         print("dataset only containes one sensitive feature, retry sampling")
            #     sub_dataloader = DataLoader(sub_dataset,
            #                             batch_size=len(sub_dataset), 
            #                             shuffle = args.shuffle,
            #                             num_workers = args.num_workers,
            #                             pin_memory = True,
            #                             drop_last = args.drop_last)
            #     return sub_dataloader
            for i in range(len(client_train_loads)):
                client_train_loads[i], _ = cut_loader(client_train_loads[i])
            for i in range(len(client_test_loads)):
                client_test_loads[i], _ = cut_loader(client_test_loads[i])
        else:
            another_loads = []
            for i in range(len(client_train_loads)):
                _, another_train = cut_loader(client_train_loads[i])
                _, another_test = cut_loader(client_test_loads[i])
                another_dataset = ConcatDataset([another_train.dataset, another_test.dataset])
                another_loader = DataLoader(another_dataset, 
                                            batch_size=len(another_dataset),
                                            shuffle = args.shuffle,
                                            num_workers = args.num_workers,
                                            pin_memory = True,
                                            drop_last = args.drop_last)
                another_loads.append(another_loader)
            client_test_loads = another_loads

    return client_train_loads, client_test_loads

