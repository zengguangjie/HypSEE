import torch
import argparse
from datasets.tu_dataset import get_dataset, get_dataset_dense
import random
import os
import numpy as np
from torch_geometric.loader import DataLoader, DenseDataLoader
from datasets.loaders import IterLoader
from models.model import HypSEE
from datasets.data_utils import hypergraph_construction, hypergraph_construction_batch
from datasets.data_utils import load_hypergraphs, hypergraph_to_dense_batch
import torch.nn.functional as F
from copy import deepcopy
from datasets.data_utils import k_fold

class ExpAug:
    def __init__(self, configs):
        self.configs = configs

    def fix_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train(self, model, labeled_loader_S, labeled_loader_T, unlabeled_loader_S, unlabeled_loader_T, optimizer,
              anchor_queue, epoch, device):
        model.train()

        labeled_loader_S.new_epoch()
        labeled_loader_T.new_epoch()
        unlabeled_loader_S.new_epoch()
        unlabeled_loader_T.new_epoch()

        total_loss = 0
        total_loss_sup = 0
        total_loss_con = 0
        total_loss_se = 0
        for batch_index in range(len(unlabeled_loader_S)):
            labeled_batch_S = labeled_loader_S.next().to(device)
            labeled_batch_T = labeled_loader_T.next().to(device)
            unlabeled_batch_S = unlabeled_loader_S.next().to(device)
            unlabeled_batch_T = unlabeled_loader_T.next().to(device)
            labeled_H_S = hypergraph_construction_batch(labeled_batch_S, self.configs['num_edges1'])
            unlabeled_H_S = hypergraph_construction_batch(unlabeled_batch_S, self.configs['num_edges2'])
            # unlabeled_graph_id_list = unlabeled_batch_S.graph_id.tolist()
            # labeled_graph_id_list = labeled_batch_S.graph_id.tolist()
            # unlabeled_H_S = hypergraph_to_dense_batch(hypergraph_dict=hypergraph_hie_aware_dict, graph_id_list=unlabeled_graph_id_list,
            #                                           num_edges=self.configs['num_edges1']).to(labeled_batch_S.x.dtype).to(device)
            # labeled_H_S = hypergraph_to_dense_batch(hypergraph_dict=hypergraph_hie_aware_dict, graph_id_list=labeled_graph_id_list,
            #                                         num_edges=self.configs['num_edges1']).to(labeled_batch_S.x.dtype).to(device)

            optimizer.zero_grad()

            S_unlabeled, T_unlabeled, out_unlabeled, S_loss_se_unlabeled, T_loss_hse_unlabeled = model(
                unlabeled_batch_S, unlabeled_batch_T, unlabeled_H_S)
            assert len(anchor_queue) == self.configs["num_anchors"]
            S_anchors = torch.stack([x[0] for x in anchor_queue])
            T_anchors = torch.stack([x[1] for x in anchor_queue])
            loss_con = model.loss_con(S_unlabeled, S_anchors, T_unlabeled, T_anchors)

            S_labeled, T_labeled, out_labeled, S_loss_se_labeled, T_loss_hse_labeled = model(
                labeled_batch_S, labeled_batch_T, labeled_H_S)
            loss_sup = F.cross_entropy(out_labeled, labeled_batch_S.y)

            if epoch > self.configs["warm_epochs"]:
                loss = loss_sup \
                       + self.configs["beta"] * loss_con \
                       + self.configs["weight_hse"] * (S_loss_se_unlabeled + T_loss_hse_unlabeled) \
                       + self.configs["weight_hse"] * (S_loss_se_labeled + T_loss_hse_labeled) / 5 \

            else:
                loss = loss_sup

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * unlabeled_batch_S.num_graphs
            total_loss_sup += float(loss_sup) * unlabeled_batch_S.num_graphs
            total_loss_con += float(
                self.configs["beta"] * loss_con) * unlabeled_batch_S.num_graphs  # unlabeled_batch.num_graphs or labeled????
            total_loss_se += float(
                self.configs["weight_hse"] * (S_loss_se_labeled + T_loss_hse_labeled)) * unlabeled_batch_S.num_graphs \
                             + float(
                self.configs["weight_hse"] * (S_loss_se_unlabeled + T_loss_hse_unlabeled)) * unlabeled_batch_S.num_graphs

            for index in range(len(labeled_batch_S)):
                anchor_queue.append((S_labeled[index, :].detach(), T_labeled[index, :].detach()))
                if len(anchor_queue) > self.configs["num_anchors"]:
                    anchor_queue.pop(0)

        return total_loss / len(unlabeled_loader_S.loader.dataset), total_loss_sup / len(
            unlabeled_loader_S.loader.dataset), \
               total_loss_con / len(unlabeled_loader_S.loader.dataset), total_loss_se / (
                   len(unlabeled_loader_S.loader.dataset))

    @torch.no_grad()
    def test(self, model, test_loader_S, test_loader_T, device):
        model.eval()

        total_correct = 0
        total_loss_sup = 0
        total_loss_hse = 0
        test_loader_S.new_epoch()
        test_loader_T.new_epoch()
        for _ in range(len(test_loader_S)):
            # for data_S, data_T in zip(test_loader_S, test_loader_T):
            test_batch_S = test_loader_S.next().to(device)
            test_batch_T = test_loader_T.next().to(device)
            H_S = hypergraph_construction_batch(test_batch_S, self.configs['num_edges1'])
            # graph_id_list = test_batch_S.graph_id.tolist()
            # H_S = hypergraph_to_dense_batch(hypergraph_dict=hypergraph_hie_aware_dict, graph_id_list=graph_id_list,
            #                                           num_edges=self.configs['num_edges1']).to(test_batch_S.x.dtype).to(device)
            S_test, T_test, out, S_loss_se_test, T_loss_hse_test = model(test_batch_S, test_batch_T, H_S)
            loss_sup = F.cross_entropy(out, test_batch_S.y)
            total_loss_sup += float(loss_sup) * test_batch_S.num_graphs
            total_loss_hse += float(self.configs["weight_hse"] * (S_loss_se_test + T_loss_hse_test)) * test_batch_S.num_graphs
            out = torch.softmax(out, dim=-1)
            pred = out.argmax(dim=-1)
            total_correct += int((pred == test_batch_S.y).sum())
        test_acc = total_correct / len(test_loader_S.loader.dataset)
        loss_sup = total_loss_sup / len(test_loader_S.loader.dataset)
        loss_se = total_loss_hse / len(test_loader_S.loader.dataset)
        return test_acc, loss_sup, loss_se

    def run(self, seed, device):
        self.fix_seed(seed)


        dataset = get_dataset(name=self.configs["data_name"], root=self.configs["data_root"],
                              feat_str=self.configs["feat_str"])
        dataset.aug = 'none'
        dataset = dataset.shuffle()
        dataset_S = deepcopy(dataset)
        dataset_T = deepcopy(dataset)

        avg_num_nodes = int(dataset._data.x.size(0) / len(dataset))

        dataset_S.aug, dataset_S.aug_ratio = self.configs["aug1"], self.configs["aug_ratio1"]
        dataset_T.aug, dataset_T.aug_ratio = self.configs["aug2"], self.configs["aug_ratio2"]

        labeled_loader_S = IterLoader(
            DataLoader(dataset_S[:0.1], batch_size=int(np.ceil(self.configs["batch_size"] / 5)), shuffle=False))
        unlabeled_loader_S = IterLoader(DataLoader(dataset_S[0.2:0.7], batch_size=self.configs["batch_size"], shuffle=False))
        labeled_loader_T = IterLoader(
            DataLoader(dataset_T[:0.1], batch_size=int(np.ceil(self.configs["batch_size"] / 5)), shuffle=False))
        unlabeled_loader_T = IterLoader(DataLoader(dataset_T[0.2:0.7], batch_size=self.configs["batch_size"], shuffle=False))

        val_loader_S = IterLoader(DataLoader(dataset[0.7:0.8], batch_size=self.configs["batch_size"], shuffle=False))
        test_loader_S = IterLoader(DataLoader(dataset[0.8:1.0], batch_size=self.configs["batch_size"], shuffle=False))
        val_loader_T = IterLoader(DataLoader(dataset[0.7:0.8], batch_size=self.configs["batch_size"], shuffle=False))
        test_loader_T = IterLoader(DataLoader(dataset[0.8:1.0], batch_size=self.configs["batch_size"], shuffle=False))

        model = HypSEE(in_channels=dataset.num_features,
                        hidden_channels_gnn=self.configs["dim_embedding_gnn"],
                        hidden_channels=self.configs["dim_embedding"],
                        out_channels=dataset.num_classes,
                        num_layers_gnn=self.configs["num_layers_gnn"],
                        num_edges2=self.configs["num_edges2"],
                        avg_num_nodes=avg_num_nodes,
                        height=self.configs["height"],
                        EPS=self.configs["EPS"]).to(device)
        for key in model.hyper_hierarchical_GRL.hyperconv_dict.keys():
            model.hyper_hierarchical_GRL.hyperconv_dict[key].to(device)
        for key in model.hyper_hierarchical_GRL.pool_dict.keys():
            model.hyper_hierarchical_GRL.pool_dict[key].to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs["lr"], weight_decay=self.configs["weight_decay"])
        model.reset_parameters()

        best_epoch = 0
        best_val_loss_sup = 1e10
        best_val_loss_sup_se = 1e10
        best_val_acc = 0.0
        best_test_acc = 0.0

        # hypergraph_hie_aware_dict = load_hypergraphs(data_name=self.configs['data_name'],
        #                                              data_root=self.configs['data_root'], mode=self.configs['mode'],
        #                                              hyperedge_length_list=self.configs['hypergraph_length_list'],
        #                                              num_edges=self.configs['num_edges1'], num_graphs=len(dataset))

        with torch.no_grad():
            labeled_loader_S.new_epoch()
            labeled_loader_T.new_epoch()
            anchor_queue = []

            while len(anchor_queue) < self.configs["num_anchors"]:
                labeled_batch_S = labeled_loader_S.next().to(device)
                labeled_batch_T = labeled_loader_T.next().to(device)
                labeled_H_S = hypergraph_construction_batch(labeled_batch_S, self.configs['num_edges1'])
                # graph_id_list = labeled_batch_S.graph_id.tolist()
                # labeled_H_S = hypergraph_to_dense_batch(hypergraph_dict=hypergraph_hie_aware_dict, graph_id_list=graph_id_list,
                #                                         num_edges=self.configs['num_edges1']).to(labeled_batch_S.x.dtype).to(device)
                S_labeled, T_labeled, out_labeled, S_loss_se_labeled, T_loss_hse_labeled = model(
                    labeled_batch_S, labeled_batch_T, labeled_H_S)
                S_labeled, T_labeled = S_labeled.detach(), T_labeled.detach()
                for i in range(S_labeled.shape[0]):
                    anchor_queue.append((S_labeled[i], T_labeled[i]))
                    if len(anchor_queue) >= self.configs["num_anchors"]:
                        break
            labeled_loader_S.new_epoch()
            labeled_loader_T.new_epoch()

        print("------------------anchors ends-----------------")

        for epoch in range(1, self.configs["epochs"] + 1):
            train_loss, train_loss_sup, train_loss_con, train_loss_se = self.train(model, labeled_loader_S, labeled_loader_T,
                                                                              unlabeled_loader_S, unlabeled_loader_T,
                                                                              optimizer, anchor_queue, epoch, device)
            val_acc, val_loss_sup, val_loss_se = self.test(model, val_loader_S, val_loader_T, device)
            test_acc, _, _ = self.test(model, test_loader_S, test_loader_T, device)
            print(
                "epoch{}:\ttrain loss: {}\ttrain_sup loss: {}\ttrain_con loss: {}\ttrain_hse loss: {}\tval_acc: {}\ttest_acc:{}"
                .format(epoch, train_loss, train_loss_sup, train_loss_con, train_loss_se, val_acc, test_acc))
            if epoch > self.configs["warm_epochs"]:
                if self.configs["epoch_select"] == 'val_loss_sup':
                    if val_loss_sup < best_val_loss_sup:
                        best_val_loss_sup = val_loss_sup
                        best_epoch = epoch
                        best_test_acc = test_acc
                        print("epoch{}:\ttest_acc:{}\tval_loss_sup\t{}".format(best_epoch, test_acc, val_loss_sup))
                elif self.configs["epoch_select"] == 'val_acc':
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch
                        best_test_acc = test_acc
                        print("epoch{}:\ttest_acc:{}".format(best_epoch, test_acc))
                elif self.configs["epoch_select"] == 'val_acc_eq':
                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch
                        best_test_acc = test_acc
                        print("epoch{}:\ttest_acc:{}".format(best_epoch, test_acc))
                elif self.configs["epoch_select"] == 'val_loss_sup_hse':
                    if val_loss_sup + val_loss_se < best_val_loss_sup_se:
                        best_val_loss_sup_se = val_loss_sup + val_loss_se
                        best_epoch = epoch
                        best_test_acc = test_acc
                        print("epoch{}:\ttest_acc:{}".format(best_epoch, test_acc))
                else:
                    raise NotImplementedError
        return best_test_acc






    def exp(self):
        # print("running")
        print(self.configs)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # self.configs["device"] = device

        print(self.configs["data_name"], self.configs["data_root"])

        self.configs['hypergraph_length_list'] = [4, 6]
        # self.configs['feat_str'] = 'deg+odeg100'
        # if self.configs['data_name'] in ['DD','REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        #     self.configs['feat_str'] = 'deg+odeg10'
        if self.configs['data_name'] == 'COLLAB':
            if self.configs['feat_str'] == '':
                self.configs['feat_str'] = 'deg+odeg100'

        test_acc_list = []
        repeat = 0
        seed = 3
        while repeat < self.configs["runs"]:
            # print(repeat)
            # with torch.autograd.detect_anomaly(True):
            try:
                best_acc = self.run(repeat, device)
                print("run: {}\tacc: {}\t".format(repeat, best_acc))
                test_acc_list.append(best_acc)
                repeat += 1
                seed += 1
            except AssertionError as e:
                seed += 1
                print(f"Caught an assertion error: {e}")
                continue
        print("test_acc_mean: {}\tstd{}".format(np.mean(test_acc_list), np.std(test_acc_list)))

        import json
        from datetime import datetime

        self.configs.mean = np.mean(test_acc_list)
        self.configs.std = np.std(test_acc_list)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"./results/{self.configs.data_name}/Time_{current_time}_ACC_{self.configs.mean}.txt"
        with open(file_name, 'w') as file:
            json.dump(self.configs, file, indent=4)

        # self.configs.test_acc_list = test_acc_list

        return np.mean(test_acc_list)
