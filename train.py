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
parser.add_argument('--warm_epochs', type=int, default=2)
parser.add_argument('--num_edges1', type=int, default=32, help='number of hyperedges per handcrafted hypergraph')
parser.add_argument('--mode', type=str, default='RW', choices=['RW', 'HOP'])
# parser.add_argument('--dense', type=bool, default=False, help='whether to use dense implementation for gnn and hgnn from the beginning')
parser.add_argument('--epoch_select', type=str, default='val_loss_sup', choices=['val_acc', 'val_loss_sup', 'val_loss_sup_hse'])
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--feat_str', type=str, default='deg')
parser.add_argument('--height', type=int, default=3)
parser.add_argument('--hypergraph_length_list', type=int, default=None)
parser.add_argument('--decay_rate', type=float, default=0.5)
args = parser.parse_args()

# args.hypergraph_length_list = [4,6]
# args.hypergraph_length_list = [2,4]

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train(model, labeled_loader, unlabeled_loader, hypergraph_hie_aware_dict, optimizer, anchor_queue, epoch, args):
    model.train()

    labeled_loader.new_epoch()
    unlabeled_loader.new_epoch()

    total_loss = 0
    total_loss_sup = 0
    total_loss_con = 0
    total_loss_se = 0
    for batch_index in range(len(unlabeled_loader)):
        labeled_batch = labeled_loader.next().to(device)
        unlabeled_batch = unlabeled_loader.next().to(device)
        labeled_bs, unlabeled_bs = torch.max(labeled_batch.batch), torch.max(unlabeled_batch.batch)

        # unlabeled_H_S, _ = hypergraph_construction_batch(unlabeled_batch, num_edges1, dense=dense)
        # labeled_H_S, _ = hypergraph_construction_batch(labeled_batch, num_edges1, dense=dense)
        # unlabeled_H_S_sparse = hypergraph_construction_batch(unlabeled_batch, num_edges1, dense=False)
        # labeled_H_S_sparse = hypergraph_construction_batch(labeled_batch, num_edges1, dense=False)

        optimizer.zero_grad()
        # print("hypergraph_construction starts")
        # unlabeled_H_S = hypergraph_construction_batch(unlabeled_batch, num_edges1, dense=True).to(labeled_batch.x.dtype)
        # labeled_H_S = hypergraph_construction_batch(labeled_batch, num_edges1, dense=True).to(labeled_batch.x.dtype)
        unlabeled_graph_id_list = unlabeled_batch.graph_id.tolist()
        labeled_graph_id_list = labeled_batch.graph_id.tolist()
        unlabeled_H_S = hypergraph_to_dense_batch(hypergraph_dict=hypergraph_hie_aware_dict, graph_id_list=unlabeled_graph_id_list, num_edges=args.num_edges1).to(labeled_batch.x.dtype).to(device)
        labeled_H_S = hypergraph_to_dense_batch(hypergraph_dict=hypergraph_hie_aware_dict, graph_id_list=labeled_graph_id_list, num_edges=args.num_edges1).to(labeled_batch.x.dtype).to(device)
        # print("hypergraph_construction ends")
        # S_unlabeled, T_unlabeled, out_unlabeled, S_loss_hse_unlabeled, T_loss_hse_unlabeled = model(unlabeled_batch.x, unlabeled_batch.edge_index, unlabeled_H_S, unlabeled_batch.batch)
        S_unlabeled, T_unlabeled, out_unlabeled, S_loss_hse_unlabeled, T_loss_hse_unlabeled = model(unlabeled_batch,
                                                                                                    unlabeled_batch,
                                                                                                    unlabeled_H_S,
                                                                                                    )

        assert len(anchor_queue) == args.num_anchors
        S_anchors = torch.stack([x[0] for x in anchor_queue])
        T_anchors = torch.stack([x[1] for x in anchor_queue])
        loss_con = model.loss_con(S_unlabeled, S_anchors, T_unlabeled, T_anchors)

        # S_labeled, T_labeled, out_labeled, S_loss_hse_labeled, T_loss_hse_labeled = model(labeled_batch.x, labeled_batch.edge_index, labeled_H_S, labeled_batch.batch)
        S_labeled, T_labeled, out_labeled, S_loss_hse_labeled, T_loss_hse_labeled = model(labeled_batch,
                                                                                          labeled_batch,
                                                                                          labeled_H_S,
                                                                                          )
        loss_sup = F.cross_entropy(out_labeled, labeled_batch.y)

        if epoch > args.warm_epochs:
            # loss = loss_sup + args.beta * loss_con + args.weight_hse * (S_loss_hse_unlabeled + T_loss_hse_unlabeled + S_loss_hse_labeled + T_loss_hse_labeled)
            loss = loss_sup + args.beta * loss_con + args.weight_hse * (S_loss_hse_unlabeled + T_loss_hse_unlabeled) + args.weight_hse * (S_loss_hse_labeled + T_loss_hse_labeled) / 5
        else:
            loss = loss_sup

        # print("batch_index{}:\ttrain loss: {}\ttrain_sup loss: {}\ttrain_con loss: {}\tS_hse_unlabeled loss: {}\tT_hse_unlabeled loss: {}\tS_hse_labeled loss: {}\tT_hse_labeled loss: {}"
        #     .format(batch_index, loss, loss_sup, loss_con, S_loss_hse_unlabeled, T_loss_hse_unlabeled, S_loss_hse_labeled, T_loss_hse_labeled))
        # print("batch_index{}:\ttrain loss: {}\ttrain_sup loss: {}\ttrain_con loss: {}\thse_unlabeled loss: {}\thse_labeled loss: {}"
        #     .format(batch_index, loss, loss_sup, loss_con, S_loss_hse_unlabeled+T_loss_hse_unlabeled,
        #             S_loss_hse_labeled+T_loss_hse_labeled))

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * unlabeled_batch.num_graphs
        total_loss_sup += float(loss_sup) * unlabeled_batch.num_graphs
        total_loss_con += float(args.beta * loss_con) * unlabeled_batch.num_graphs   # unlabeled_batch.num_graphs or labeled????
        total_loss_se += float(args.weight_hse * (S_loss_hse_labeled + T_loss_hse_labeled))  * unlabeled_batch.num_graphs \
                         + float(args.weight_hse * (S_loss_hse_unlabeled + T_loss_hse_unlabeled)) * unlabeled_batch.num_graphs

        for index in range(labeled_bs):
            anchor_queue.append((S_labeled[index, :].detach(), T_labeled[index, :].detach()))
            if len(anchor_queue) > args.num_anchors:
                anchor_queue.pop(0)

    return total_loss / len(unlabeled_loader.loader.dataset), total_loss_sup / len(unlabeled_loader.loader.dataset), \
           total_loss_con / len(unlabeled_loader.loader.dataset), total_loss_se / (len(unlabeled_loader.loader.dataset))


@torch.no_grad()
def test_noloss(model, test_loader, hypergraph_hie_aware_dict, num_edges1):
    model.eval()

    total_correct = 0
    test_loader.new_epoch()
    for _ in range(len(test_loader)):
        val_batch = test_loader.next().to(device)
        val_graph_id_list = val_batch.graph_id.tolist()
        val_H_S = hypergraph_to_dense_batch(hypergraph_dict=hypergraph_hie_aware_dict,
                                                graph_id_list=val_graph_id_list, num_edges=num_edges1).to(val_batch.x.dtype).to(device)
        # val_H_S = hypergraph_construction_batch(val_batch, num_edges1).to(val_batch.x.dtype)
        S_val, T_val, out, _, _ = model(val_batch.x, val_batch.edge_index, val_H_S, val_batch.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == val_batch.y).sum())
    return total_correct / len(test_loader.loader.dataset)

@torch.no_grad()
def test(model, test_loader, hypergraph_hie_aware_dict, args):
    model.eval()

    total_correct = 0
    total_loss_sup = 0
    total_loss_hse = 0
    test_loader.new_epoch()
    for _ in range(len(test_loader)):
        test_batch = test_loader.next().to(device)
        test_graph_id_list = test_batch.graph_id.tolist()
        test_H_S = hypergraph_to_dense_batch(hypergraph_dict=hypergraph_hie_aware_dict,
                                            graph_id_list=test_graph_id_list, num_edges=args.num_edges1).to(test_batch.x.dtype).to(device)
        # S_test, T_test, out, S_loss_hse_test, T_loss_hse_test = model(test_batch.x, test_batch.edge_index, test_H_S, test_batch.batch)
        S_test, T_test, out, S_loss_hse_test, T_loss_hse_test = model(test_batch, test_batch, test_H_S,
                                                                      )
        loss_sup = F.cross_entropy(out, test_batch.y)
        total_loss_sup += float(loss_sup) * test_batch.num_graphs
        total_loss_hse += float(args.weight_hse * (S_loss_hse_test + T_loss_hse_test))
        out = torch.softmax(out, dim=-1)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == test_batch.y).sum())
    test_acc = total_correct / len(test_loader.loader.dataset)
    loss_sup = total_loss_sup / len(test_loader.loader.dataset)
    loss_hse = total_loss_hse / len(test_loader.loader.dataset)
    return test_acc, loss_sup, loss_hse


def run(seed, device, args, avg_num_nodes=None):
    fix_seed(seed)
    # dataset.shuffle()
    dataset = get_dataset(name=args.data_name, root=args.data_root, feat_str=args.feat_str).shuffle()

    hypergraph_hie_aware_dict = load_hypergraphs(data_name=args.data_name, data_root=args.data_root, mode=args.mode,
                     hyperedge_length_list=args.hypergraph_length_list, num_edges=args.num_edges1, num_graphs=len(dataset))
    if avg_num_nodes is None:
        avg_num_nodes = int(dataset._data.x.size(0) / len(dataset))

    # test_acc_list = []
    # for run in range(args.runs):
        # avg_num_nodes = int(dataset._data.x.size(0) / len(dataset))
    labeled_loader = IterLoader(DataLoader(dataset[:0.1], batch_size=int(np.ceil(args.batch_size / 5)), shuffle=True))
    unlabeled_loader = IterLoader(DataLoader(dataset[0.2:0.7], batch_size=args.batch_size, shuffle=True))
    val_loader = IterLoader(DataLoader(dataset[0.7:0.8], batch_size=args.batch_size, shuffle=False))
    test_loader = IterLoader(DataLoader(dataset[0.8:1.0], batch_size=args.batch_size, shuffle=False))
    model = HypSEE(in_channels=dataset.num_features,
                   hidden_channels_gnn=args.dim_embedding_gnn,
                   hidden_channels=args.dim_embedding,
                   out_channels=dataset.num_classes,
                   num_layers_gnn=args.num_layers_gnn,
                   num_edges2=args.num_edges2,
                   avg_num_nodes=avg_num_nodes,
                   height=args.height,
                   decay_rate=args.decay_rate).to(device)
    for key in model.hyper_hierarchical_GRL.hyperconv_dict.keys():
        model.hyper_hierarchical_GRL.hyperconv_dict[key].to(device)
    for key in model.hyper_hierarchical_GRL.pool_dict.keys():
        model.hyper_hierarchical_GRL.pool_dict[key].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.reset_parameters()

    best_epoch = 0
    best_val_loss_sup = 1e10
    best_val_loss_sup_hse = 1e10
    best_val_acc = 0.0
    best_test_acc = 0.0

    with torch.no_grad():
        labeled_loader.new_epoch()
        anchor_queue = []

        while len(anchor_queue) < args.num_anchors:
            labeled_batch = labeled_loader.next().to(device)
            graph_id_list = labeled_batch.graph_id.tolist()
            labeled_H_S = hypergraph_to_dense_batch(hypergraph_dict=hypergraph_hie_aware_dict,
                                                    graph_id_list=graph_id_list, num_edges=args.num_edges1).to(labeled_batch.x.dtype).to(device)
            # labeled_H_S = hypergraph_construction_batch(labeled_batch, args.num_edges1).to(labeled_batch.x.dtype)
            # S_labeled, T_labeled, out_labeled, S_loss_hse_labeled, T_loss_hse_labeled = model(labeled_batch.x,
            #                                                                                   labeled_batch.edge_index,
            #                                                                                   labeled_H_S,
            #                                                                                   labeled_batch.batch)
            S_labeled, T_labeled, out_labeled, S_loss_hse_labeled, T_loss_hse_labeled = model(labeled_batch,
                                                                                              labeled_batch,
                                                                                              labeled_H_S,
                                                                                              )
            S_labeled, T_labeled = S_labeled.detach(), T_labeled.detach()
            for i in range(S_labeled.shape[0]):
                anchor_queue.append((S_labeled[i], T_labeled[i]))
                if len(anchor_queue) >= args.num_anchors:
                    break
        labeled_loader.new_epoch()

    print("------------------anchors ends-----------------")

    for epoch in range(1, args.epochs+1):
        train_loss, train_loss_sup, train_loss_con, train_loss_hse = train(model, labeled_loader, unlabeled_loader,
                                                                           hypergraph_hie_aware_dict, optimizer,
                                                                           anchor_queue, epoch, args)
        val_acc, val_loss_sup, val_loss_hse = test(model, val_loader, hypergraph_hie_aware_dict, args)
        test_acc, _, _ = test(model, test_loader, hypergraph_hie_aware_dict, args)
        print("epoch{}:\ttrain loss: {}\ttrain_sup loss: {}\ttrain_con loss: {}\ttrain_hse loss: {}\tval_acc: {}\ttest_acc:{}"
              .format(epoch, train_loss, train_loss_sup, train_loss_con, train_loss_hse, val_acc, test_acc))
        if epoch > args.warm_epochs:
            if args.epoch_select == 'val_loss_sup':
                if val_loss_sup < best_val_loss_sup:
                    best_val_loss_sup = val_loss_sup
                    best_epoch = epoch
                    best_test_acc = test_acc
                    print("epoch{}:\ttest_acc:{}".format(best_epoch, test_acc))
            elif args.epoch_select == 'val_acc':
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_test_acc = test_acc
                    print("epoch{}:\ttest_acc:{}".format(best_epoch, test_acc))
            elif args.epoch_select == 'val_loss_sup_hse':
                if val_loss_sup + val_loss_hse < best_val_loss_sup_hse:
                    best_val_loss_sup_hse = val_loss_sup + val_loss_hse
                    best_epoch = epoch
                    best_test_acc = test_acc
                    print("epoch{}:\ttest_acc:{}".format(best_epoch, test_acc))
            else:
                raise NotImplementedError

        # test_acc_list.append(best_test_acc)
    return best_test_acc




if __name__=='__main__':
    if args.hypergraph_length_list is None:
        if args.mode == 'RW':
            if args.data_name in ['IMDB-BINARY', 'IMDB-MULTI']:
                args.hypergraph_length_list = [4, 6]
            else:
                args.hypergraph_length_list = [4, 6]
        elif args.mode == 'HOP':
            if args.data_name in ['IMDB-BINARY', 'IMDB-MULTI']:
                args.hypergraph_length_list = [1,2]
            else:
                args.hypergraph_length_list = [2,4]

    args.data_root = os.path.join(args.data_root, args.data_name)

    args.feat_str = ''

    # args.feat_str = 'deg+odeg100'
    # if args.data_name in ['DD','REDDIT-BINARY', 'REDDIT-MULTI-5K']:
    #     args.feat_str = 'deg+odeg10'



    avg_num_nodes = None
    # if args.data_name == 'COLLAB':
    #     avg_num_nodes = 74
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset = get_dataset(name=args.data_name, root=args.data_root, feat_str=args.feat_str).shuffle()
    #
    # hypergraph_hie_aware_dict = load_hypergraphs(data_name=args.data_name, data_root=args.data_root, mode=args.mode,
    #                  hyperedge_length_list=args.hypergraph_length_list, num_edges=args.num_edges1, num_graphs=len(dataset))

    test_acc_list = []
    repeat = 0
    seed = 0
    while repeat < args.runs:
    # for repeat in range(args.runs):
        try:
            # best_acc = run(dataset, hypergraph_hie_aware_dict, seed, device, args, avg_num_nodes)
            best_acc = run(seed, device, args, avg_num_nodes)
            print("run: {}\tacc: {}\t".format(repeat, best_acc))
            test_acc_list.append(best_acc)
            repeat += 1
            seed += 1
        except AssertionError as e:
            seed += 1
            print(f"Caught an assertion error: {e}")
            continue

    # seed = 0
    # test_acc_list = cross_validation(dataset=dataset,
    #                  hypergraph_hie_aware_dict=hypergraph_hie_aware_dict,
    #                  seed=seed,
    #                  device=device,
    #                  args=args)
    # test_acc_list = test_acc_list[:args.runs]
    print("test_acc_mean: {}\tstd{}".format(np.mean(test_acc_list), np.std(test_acc_list)))









    # fix_seed(4)
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    #
    #
    #
    # dataset = get_dataset(name=args.data_name, root=args.data_root).shuffle()
    #
    # hypergraph_hie_aware_dict = load_hypergraphs(data_name=args.data_name, data_root=args.data_root, mode=args.mode,
    #                  hyperedge_length_list=args.hypergraph_length_list, num_edges=args.num_edges1, num_graphs=len(dataset))
    #
    # avg_num_nodes = int(dataset._data.x.size(0) / len(dataset))
    # labeled_loader = IterLoader(DataLoader(dataset[:0.1], batch_size=int(np.ceil(args.batch_size / 5)), shuffle=True))
    # unlabeled_loader = IterLoader(DataLoader(dataset[0.2:0.7], batch_size=args.batch_size, shuffle=True))
    # val_loader = IterLoader(DataLoader(dataset[0.7:0.8], batch_size=args.batch_size, shuffle=True))
    # test_loader = IterLoader(DataLoader(dataset[0.8:1.0], batch_size=args.batch_size, shuffle=False))
    # model = HypSEE(in_channels=dataset.num_features,
    #                hidden_channels_gnn=args.dim_embedding_gnn,
    #                hidden_channels=args.dim_embedding,
    #                out_channels=dataset.num_classes,
    #                num_layers_gnn=args.num_layers_gnn,
    #                num_edges2=args.num_edges2,
    #                avg_num_nodes=avg_num_nodes).to(device)
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # model.reset_parameters()
    #
    # best_val = 0.0
    # best_epoch = 0
    #
    # # with torch.autograd.detect_anomaly(True):
    # with torch.no_grad():
    #     labeled_loader.new_epoch()
    #     anchor_queue = []
    #
    #     while len(anchor_queue) < args.num_anchors:
    #         labeled_batch = labeled_loader.next().to(device)
    #         graph_id_list = labeled_batch.graph_id.tolist()
    #         labeled_H_S = hypergraph_to_dense_batch(hypergraph_dict=hypergraph_hie_aware_dict, graph_id_list=graph_id_list, num_edges=args.num_edges1).to(labeled_batch.x.dtype).to(device)
    #         # labeled_H_S = hypergraph_construction_batch(labeled_batch, args.num_edges1).to(labeled_batch.x.dtype)
    #         S_labeled, T_labeled, out_labeled, S_loss_hse_labeled, T_loss_hse_labeled = model(labeled_batch.x, labeled_batch.edge_index, labeled_H_S, labeled_batch.batch)
    #         S_labeled, T_labeled = S_labeled.detach(), T_labeled.detach()
    #         for i in range(S_labeled.shape[0]):
    #             anchor_queue.append((S_labeled[i], T_labeled[i]))
    #             if len(anchor_queue) >= args.num_anchors:
    #                 break
    #     labeled_loader.new_epoch()
    #
    # print("------------------anchors ends-----------------")
    #
    # # with torch.autograd.detect_anomaly(True):
    # final_acc = 0
    # for epoch in range(1, args.epochs + 1):
    #     train_loss, train_loss_sup, train_loss_con, train_loss_hse = train(model, labeled_loader, unlabeled_loader, hypergraph_hie_aware_dict, optimizer, anchor_queue, epoch, args)
    #     # exit(0)
    #     val_acc = test(model, val_loader, hypergraph_hie_aware_dict, args)
    #     test_acc = test(model, test_loader, hypergraph_hie_aware_dict, args)
    #     # val_acc = 0
    #     print("epoch{}:\ttrain loss: {}\ttrain_sup loss: {}\ttrain_con loss: {}\ttrain_hse loss: {}\tval_acc: {}\ttest_acc:{}".format(epoch, train_loss, train_loss_sup, train_loss_con, train_loss_hse, val_acc, test_acc))
    #     if epoch > args.warm_epochs:
    #         if val_acc > best_val:
    #             best_val = val_acc
    #             best_epoch = epoch
    #             print("epoch{}:\ttest_acc:{}".format(epoch, test_acc))
    #             final_acc = test_acc
    #         # torch.save(model.state_dict(), './models/' + args.data_name + '_' + str(best_epoch) + '.pt')
    #
    # # model.load_state_dict(torch.load('./models/' + args.data_name + '_' + str(best_epoch) + '.pt'))
    # # test_acc = test(model, test_loader, hypergraph_hie_aware_dict, args.num_edges1)
    # print(final_acc)