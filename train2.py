# train for HypSEE2

import torch
import argparse
from datasets.tu_dataset import get_dataset, get_dataset_dense
import random
import os
import numpy as np
from torch_geometric.loader import DataLoader, DenseDataLoader
from datasets.loaders import IterLoader
from models.model import HypSEE2
from datasets.data_utils import hypergraph_construction, hypergraph_construction_batch
from datasets.data_utils import load_hypergraphs, hypergraph_to_dense_batch
import torch.nn.functional as F
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='IMDB-BINARY')
parser.add_argument('--data_root', type=str, default='data/')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
# parser.add_argument('--lr_decay_step_size', type=int, default=1)
parser.add_argument('--dim_embedding', type=int, default=32)
parser.add_argument('--dim_embedding_gnn', type=int, default=0)
parser.add_argument('--num_edges2', type=int, default=32)
parser.add_argument('--num_layers_gnn', type=int, default=2, help='number of layers in the first encoder embedding.')
parser.add_argument('--num_anchors', type=int, default=64, help='number of anchor graphs to be used')
parser.add_argument('--beta', type=float, default=1, help='beta balances two loss values.')
parser.add_argument('--weight_hse', type=float, default=0.001, help="weight of hierarchical se loss")
parser.add_argument('--warm_epochs', type=int, default=5)
# parser.add_argument('--num_edges1', type=int, default=32, help='number of hyperedges per handcrafted hypergraph')
# parser.add_argument('--mode', type=str, default='RW', choices=['RW', 'HOP'])
# parser.add_argument('--dense', type=bool, default=False, help='whether to use dense implementation for gnn and hgnn from the beginning')
parser.add_argument('--epoch_select', type=str, default='val_loss_sup', choices=['val_acc', 'val_loss_sup', 'val_loss_sup_hse', 'val_acc_eq'])
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--feat_str', type=str, default='deg')
parser.add_argument('--height', type=int, default=3)
# parser.add_argument('--hypergraph_length_list', type=int, default=None)
parser.add_argument('--T2', type=float, default=1.0, help='temperature for fix_match loss')
parser.add_argument('--threshold', type=float, default=0.95, help='threshold for fix_match loss')
parser.add_argument('--weight_fix', type=float, default=1)
parser.add_argument('--weight_simlr', type=float, default=1)
parser.add_argument('--aug1', type=str, default='random2', choices=['dropN', 'wdropN', 'permE', 'subgraph', 'maskN', 'none', 'random4', 'random3', 'random2'])
parser.add_argument('--aug_ratio1', type=float, default=0.2)
parser.add_argument('--aug2', type=str, default='random2', choices=['dropN', 'wdropN', 'permE', 'subgraph', 'maskN', 'none', 'random4', 'random3', 'random2'])
parser.add_argument('--aug_ratio2', type=float, default=0.2)
args = parser.parse_args()

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train(model, labeled_loader_S, labeled_loader_T, unlabeled_loader_S, unlabeled_loader_T, optimizer, anchor_queue, epoch, device, args):
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

        optimizer.zero_grad()

        S_unlabeled, T_unlabeled, S_out_unlabeled, T_out_unlabeled, S_loss_se_unlabeled, T_loss_hse_unlabeled = model(unlabeled_batch_S, unlabeled_batch_T)
        assert len(anchor_queue) == args.num_anchors
        S_anchors = torch.stack([x[0] for x in anchor_queue])
        T_anchors = torch.stack([x[1] for x in anchor_queue])
        loss_con = model.loss_con(S_unlabeled, S_anchors, T_unlabeled, T_anchors)
        loss_fix = model.loss_fix(S_out_unlabeled, T_out_unlabeled, args.T2, args.threshold)
        loss_simlr = model.loss_simlr(S_unlabeled, T_unlabeled)

        S_labeled, T_labeled, S_out_labeled, T_out_labeled, S_loss_se_labeled, T_loss_hse_labeled = model(labeled_batch_S, labeled_batch_T)
        loss_sup = F.cross_entropy(S_out_labeled, labeled_batch_S.y)

        if epoch > args.warm_epochs:
            loss = loss_sup \
                   + args.beta * loss_con \
                + args.weight_hse * (S_loss_se_unlabeled + T_loss_hse_unlabeled) \
                    + args.weight_hse * (S_loss_se_labeled + T_loss_hse_labeled) / 5 \
                   + args.weight_fix * loss_fix \
                   + args.weight_simlr * loss_simlr

        else:
            loss = loss_sup
        # print("batch_index{}:\ttrain loss: {}\ttrain_sup loss: {}\ttrain_con loss: {}\thse_unlabeled loss: {}\thse_labeled loss: {}"
        #     .format(batch_index, loss, loss_sup, loss_con, S_loss_se_unlabeled+T_loss_hse_unlabeled,
        #             S_loss_se_labeled+T_loss_hse_labeled))

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * unlabeled_batch_S.num_graphs
        total_loss_sup += float(loss_sup) * unlabeled_batch_S.num_graphs
        total_loss_con += float(args.beta * loss_con) * unlabeled_batch_S.num_graphs   # unlabeled_batch.num_graphs or labeled????
        total_loss_se += float(args.weight_hse * (S_loss_se_labeled + T_loss_hse_labeled))  * unlabeled_batch_S.num_graphs \
                         + float(args.weight_hse * (S_loss_se_unlabeled + T_loss_hse_unlabeled)) * unlabeled_batch_S.num_graphs

        for index in range(len(labeled_batch_S)):
            anchor_queue.append((S_labeled[index, :].detach(), T_labeled[index, :].detach()))
            if len(anchor_queue) > args.num_anchors:
                anchor_queue.pop(0)

    return total_loss / len(unlabeled_loader_S.loader.dataset), total_loss_sup / len(unlabeled_loader_S.loader.dataset), \
           total_loss_con / len(unlabeled_loader_S.loader.dataset), total_loss_se / (len(unlabeled_loader_S.loader.dataset))


@torch.no_grad()
def test(model, test_loader_S, test_loader_T, device, args):
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
        S_test, T_test, out_S, out_T, S_loss_se_test, T_loss_hse_test = model(test_batch_S, test_batch_T)
        loss_sup = F.cross_entropy(out_S, test_batch_S.y)
        total_loss_sup += float(loss_sup) * test_batch_S.num_graphs
        total_loss_hse += float(args.weight_hse * (S_loss_se_test + T_loss_hse_test)) * test_batch_S.num_graphs
        out_S = torch.softmax(out_S, dim=-1)
        pred = out_S.argmax(dim=-1)
        total_correct += int((pred == test_batch_S.y).sum())
    test_acc = total_correct / len(test_loader_S.loader.dataset)
    loss_sup = total_loss_sup / len(test_loader_S.loader.dataset)
    loss_se = total_loss_hse / len(test_loader_S.loader.dataset)
    return test_acc, loss_sup, loss_se


def run(dataset, seed, device, args):
    fix_seed(seed)
    avg_num_nodes = int(dataset._data.x.size(0) / len(dataset))

    dataset.aug = 'none'
    dataset_S = dataset.shuffle()
    dataset_S.aug, dataset_S.aug_ratio = args.aug1, args.aug_ratio1
    dataset_T = deepcopy(dataset_S)
    dataset_T.aug, dataset_T.aug_ratio = args.aug2, args.aug_ratio2


    labeled_loader_S = IterLoader(DataLoader(dataset_S[:0.1], batch_size=int(np.ceil(args.batch_size / 5)), shuffle=False))
    unlabeled_loader_S = IterLoader(DataLoader(dataset_S[0.2:0.7], batch_size=args.batch_size, shuffle=False))
    val_loader_S = IterLoader(DataLoader(dataset_S[0.7:0.8], batch_size=args.batch_size, shuffle=False))
    test_loader_S = IterLoader(DataLoader(dataset_S[0.8:1.0], batch_size=args.batch_size, shuffle=False))

    labeled_loader_T = IterLoader(DataLoader(dataset_T[:0.1], batch_size=int(np.ceil(args.batch_size / 5)), shuffle=False))
    unlabeled_loader_T = IterLoader(DataLoader(dataset_T[0.2:0.7], batch_size=args.batch_size, shuffle=False))
    val_loader_T = IterLoader(DataLoader(dataset_T[0.7:0.8], batch_size=args.batch_size, shuffle=False))
    test_loader_T = IterLoader(DataLoader(dataset_T[0.8:1.0], batch_size=args.batch_size, shuffle=False))

    model = HypSEE2(in_channels=dataset.num_features,
                   hidden_channels_gnn=args.dim_embedding_gnn,
                   hidden_channels=args.dim_embedding,
                   out_channels=dataset.num_classes,
                   num_layers_gnn=args.num_layers_gnn,
                   num_edges2=args.num_edges2,
                   avg_num_nodes=avg_num_nodes,
                   height=args.height).to(device)
    for key in model.hyper_hierarchical_GRL.hyperconv_dict.keys():
        model.hyper_hierarchical_GRL.hyperconv_dict[key].to(device)
    for key in model.hyper_hierarchical_GRL.pool_dict.keys():
        model.hyper_hierarchical_GRL.pool_dict[key].to(device)
    for key in model.hierarchical_GRL.conv_dict.keys():
        model.hierarchical_GRL.conv_dict[key].to(device)
    for key in model.hierarchical_GRL.pool_dict.keys():
        model.hierarchical_GRL.pool_dict[key].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.reset_parameters()

    best_epoch = 0
    best_val_loss_sup = 1e10
    best_val_loss_sup_se = 1e10
    best_val_acc = 0.0
    best_test_acc = 0.0

    with torch.no_grad():
        labeled_loader_S.new_epoch()
        labeled_loader_T.new_epoch()
        anchor_queue = []

        while len(anchor_queue) < args.num_anchors:
            labeled_batch_S = labeled_loader_S.next().to(device)
            labeled_batch_T = labeled_loader_T.next().to(device)
            S_labeled, T_labeled, S_out_labeled, T_out_labeled, S_loss_se_labeled, T_loss_hse_labeled = model(labeled_batch_S, labeled_batch_T)
            S_labeled, T_labeled = S_labeled.detach(), T_labeled.detach()
            for i in range(S_labeled.shape[0]):
                anchor_queue.append((S_labeled[i], T_labeled[i]))
                if len(anchor_queue) >= args.num_anchors:
                    break
        labeled_loader_S.new_epoch()
        labeled_loader_T.new_epoch()

    print("------------------anchors ends-----------------")

    for epoch in range(1, args.epochs+1):
        train_loss, train_loss_sup, train_loss_con, train_loss_se = train(model, labeled_loader_S, labeled_loader_T,
                                                                          unlabeled_loader_S, unlabeled_loader_T,
                                                                          optimizer, anchor_queue, epoch, device, args)
        val_acc, val_loss_sup, val_loss_se = test(model, val_loader_S, val_loader_T, device, args)
        test_acc, _, _ = test(model, test_loader_S, test_loader_T, device, args)
        print("epoch{}:\ttrain loss: {}\ttrain_sup loss: {}\ttrain_con loss: {}\ttrain_hse loss: {}\tval_acc: {}\ttest_acc:{}"
              .format(epoch, train_loss, train_loss_sup, train_loss_con, train_loss_se, val_acc, test_acc))
        if epoch > args.warm_epochs:
            if args.epoch_select == 'val_loss_sup':
                if val_loss_sup < best_val_loss_sup:
                    best_val_loss_sup = val_loss_sup
                    best_epoch = epoch
                    best_test_acc = test_acc
                    print("epoch{}:\ttest_acc:{}\tval_loss_sup\t{}".format(best_epoch, test_acc, val_loss_sup))
            elif args.epoch_select == 'val_acc':
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_test_acc = test_acc
                    print("epoch{}:\ttest_acc:{}".format(best_epoch, test_acc))
            elif args.epoch_select == 'val_loss_sup_hse':
                if val_loss_sup + val_loss_se < best_val_loss_sup_se:
                    best_val_loss_sup_se = val_loss_sup + val_loss_se
                    best_epoch = epoch
                    best_test_acc = test_acc
                    print("epoch{}:\ttest_acc:{}".format(best_epoch, test_acc))
            else:
                raise NotImplementedError
    return best_test_acc


if __name__=='__main__':
    args.data_root = os.path.join(args.data_root, args.data_name)
    args.feat_str = 'deg+odeg100'
    if args.data_name in ['DD','REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        args.feat_str = 'deg+odeg10'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = get_dataset(name=args.data_name, root=args.data_root, feat_str=args.feat_str)
    test_acc_list = []
    for repeat in range(args.runs):
        # with torch.autograd.detect_anomaly(True):
        best_acc = run(dataset, repeat, device, args)
        print("run: {}\tacc: {}\t".format(repeat, best_acc))
        test_acc_list.append(best_acc)
    print("test_acc_mean: {}\tstd{}".format(np.mean(test_acc_list), np.std(test_acc_list)))