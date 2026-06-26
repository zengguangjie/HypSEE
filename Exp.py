
import torch
import argparse
import traceback
from typing import Any, cast
from datasets.tu_dataset import get_dataset, subset_dataset, shuffle_dataset, TUDatasetExt, augment_batch
import random
import os
import numpy as np
from torch_geometric.loader import DataLoader
from datasets.loaders import IterLoader
from models.model import HypSEE
from datasets.data_utils import load_hypergraphs, hypergraph_to_dense_batch
import torch.nn.functional as F
from copy import deepcopy

class Exp:
    def __init__(self, configs):
        self.configs = configs
        self._debug_ctx: dict[str, Any] = {}
        self.wandb = None
        if configs.get("use_wandb"):
            import wandb
            if wandb.run is not None:
                self.wandb = wandb
                self.setup_wandb_metrics()

    @staticmethod
    def setup_wandb_metrics():
        import wandb
        if wandb.run is None:
            return
        wandb.define_metric("epoch", summary="none")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("eval/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("exp_iter", summary="max")
        wandb.define_metric("iter/*", step_metric="exp_iter")
        wandb.define_metric("final/*", step_metric="exp_iter")

    def _set_debug_ctx(self, **kwargs):
        self._debug_ctx.update(kwargs)

    def _diagnose_assertion(self):
        ctx = self._debug_ctx
        print("[debug] failure context:", {k: v for k, v in ctx.items() if k != "model"})
        model = ctx.get("model")
        if model is None:
            return
        bad_params = []
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                bad_params.append(
                    f"{name}(nan={int(torch.isnan(param).sum())}, inf={int(torch.isinf(param).sum())})"
                )
        if bad_params:
            print("[debug] non-finite parameters:")
            for line in bad_params:
                print(" ", line)
        else:
            print("[debug] all parameters are finite (failure likely in activations)")

    def _handle_assertion_error(self, err: AssertionError, seed: int):
        print(f"seed {seed} failed with assertion error: {err}")
        print(traceback.format_exc())
        self._diagnose_assertion()
        if self.configs.get("debug"):
            raise

    @staticmethod
    def _is_finite(*tensors: torch.Tensor) -> bool:
        return all(torch.isfinite(t).all() for t in tensors)

    def _optimizer_step(self, model, optimizer, loss):
        if not self._is_finite(loss):
            print(f"[warn] skip step: non-finite loss ({loss.item()})")
            optimizer.zero_grad(set_to_none=True)
            return False

        loss.backward()
        grad_clip = self.configs.get("grad_clip", 5.0)
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return True

    def fix_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _log_wandb_train(self, epoch, exp_iter, train_loss, train_loss_sup, train_loss_con, train_loss_hse):
        if self.wandb is None:
            return
        self.wandb.log({
            'epoch': epoch,
            f'train/loss_iter{exp_iter}': train_loss,
            f'train/loss_sup_iter{exp_iter}': train_loss_sup,
            f'train/loss_con_iter{exp_iter}': train_loss_con,
            f'train/loss_hse_iter{exp_iter}': train_loss_hse,
        })

    def _log_wandb_eval(self, epoch, exp_iter, val_acc, val_loss_sup, val_loss_hse, val_loss_con):
        if self.wandb is None:
            return
        self.wandb.log({
            'epoch': epoch,
            f'eval/acc_iter{exp_iter}': val_acc,
            f'eval/loss_sup_iter{exp_iter}': val_loss_sup,
            f'eval/loss_con_iter{exp_iter}': val_loss_con,
            f'eval/loss_hse_iter{exp_iter}': val_loss_hse,
            f'eval/loss_iter{exp_iter}': val_loss_sup + val_loss_hse + val_loss_con,
        })

    def _log_wandb_test_epoch(self, epoch, exp_iter, test_acc, test_loss_sup, test_loss_hse, test_loss_con):
        if self.wandb is None:
            return
        self.wandb.log({
            'epoch': epoch,
            f'test/acc_iter{exp_iter}': test_acc,
            f'test/loss_sup_iter{exp_iter}': test_loss_sup,
            f'test/loss_con_iter{exp_iter}': test_loss_con,
            f'test/loss_hse_iter{exp_iter}': test_loss_hse,
            f'test/loss_iter{exp_iter}': test_loss_sup + test_loss_hse + test_loss_con,
        })

    def _log_iter_metrics(self, exp_iter, test_acc, test_loss_sup, test_loss_hse, test_loss_con,
                          best_epoch, seed, test_acc_list):
        if self.wandb is None:
            return
        log_dict = {
            'exp_iter': exp_iter + 1,
            'iter/exp_iter': exp_iter + 1,
            'iter/test_acc': test_acc,
            'iter/test_loss_sup': test_loss_sup,
            'iter/test_loss_con': test_loss_con,
            'iter/test_loss_hse': test_loss_hse,
            'iter/test_loss': test_loss_sup + test_loss_hse + test_loss_con,
            'iter/best_epoch': best_epoch,
            'iter/seed': seed,
            'iter/avg_test_acc': float(np.mean(test_acc_list)),
            'final/avg_test_acc': float(np.mean(test_acc_list)),
        }
        if len(test_acc_list) > 1:
            log_dict['final/std_test_acc'] = float(np.std(test_acc_list))
        self.wandb.log(log_dict)

    def train(self, model, labeled_loader, unlabeled_loader, hypergraph_hie_aware_dict, optimizer, anchor_queue, epoch, device):
        model.train()

        labeled_loader.new_epoch()
        unlabeled_loader.new_epoch()

        assert len(anchor_queue) == self.configs["num_anchors"]

        warm = epoch <= self.configs["warm_epochs"]
        beta = self.configs["beta"]
        weight_hse = self.configs["weight_hse"]
        num_steps = len(unlabeled_loader)

        total_loss = 0.0
        total_loss_sup = 0.0
        total_loss_con = 0.0
        total_loss_hse = 0.0
        total_labeled_graphs = 0

        for _ in range(num_steps):
            labeled_batch = labeled_loader.next().to(device)
            unlabeled_batch = unlabeled_loader.next().to(device)
            num_labeled_graphs = labeled_batch.num_graphs
            num_unlabeled_graphs = unlabeled_batch.num_graphs

            labeled_H_S = self._batch_hypergraph(labeled_batch, hypergraph_hie_aware_dict, device)
            labeled_data_S, labeled_data_T = self._augment_views(labeled_batch, device)
            optimizer.zero_grad()

            S_labeled, T_labeled, out_labeled, loss_hse_S_labeled, loss_hse_T_labeled = model(
                labeled_data_S, labeled_data_T, labeled_H_S)
            loss_sup = F.cross_entropy(out_labeled, labeled_batch.y)

            unlabeled_H_S = self._batch_hypergraph(unlabeled_batch, hypergraph_hie_aware_dict, device)
            unlabeled_data_S, unlabeled_data_T = self._augment_views(unlabeled_batch, device)

            def _compute_aux_losses():
                S_unlabeled, T_unlabeled, _, loss_hse_S_unlabeled, loss_hse_T_unlabeled = model(
                    unlabeled_data_S, unlabeled_data_T, unlabeled_H_S)
                S_anchors = torch.stack([x[0] for x in anchor_queue])
                T_anchors = torch.stack([x[1] for x in anchor_queue])
                loss_con = model.loss_con(S_unlabeled, S_anchors, T_unlabeled, T_anchors)
                hse_labeled_scale = num_labeled_graphs / max(num_unlabeled_graphs, 1)
                loss_hse = (loss_hse_S_unlabeled + loss_hse_T_unlabeled
                            + hse_labeled_scale * (loss_hse_S_labeled + loss_hse_T_labeled))
                return loss_con, loss_hse

            if warm:
                with torch.no_grad():
                    loss_con, loss_hse = _compute_aux_losses()
                loss = loss_sup
            else:
                loss_con, loss_hse = _compute_aux_losses()
                loss = loss_sup + beta * loss_con + weight_hse * loss_hse

            total_loss_con += float(beta * loss_con)
            total_loss_hse += float(weight_hse * loss_hse)

            if not self._optimizer_step(model, optimizer, loss):
                continue

            total_loss += float(loss)
            total_loss_sup += float(loss_sup) * num_labeled_graphs
            total_labeled_graphs += num_labeled_graphs

            for i in range(num_labeled_graphs):
                anchor_queue.append((S_labeled[i].detach(), T_labeled[i].detach()))
                if len(anchor_queue) > self.configs["num_anchors"]:
                    anchor_queue.pop(0)

        return (
            total_loss / num_steps,
            total_loss_sup / total_labeled_graphs,
            total_loss_con / num_steps,
            total_loss_hse / num_steps,
        )

    def _batch_hypergraph(self, batch, hypergraph_hie_aware_dict, device):
        return hypergraph_to_dense_batch(
            hypergraph_dict=hypergraph_hie_aware_dict,
            graph_id_list=[int(gid) for gid in batch.graph_id.view(-1).tolist()],
            num_edges=self.configs['num_edges1'],
        ).to(batch.x.dtype).to(device)

    def _augment_views(self, batch, device, augment=True):
        if not augment:
            batch = batch.to(device)
            return batch, batch

        cfg = self.configs
        aug1 = cfg.get("aug1", "none")
        aug_ratio1 = cfg.get("aug_ratio1", 0.2)
        aug2 = cfg.get("aug2", "none")
        aug_ratio2 = cfg.get("aug_ratio2", 0.2)
        npower = cfg.get("npower", 1.0)

        data_S = cast(Any, augment_batch(batch, aug1, aug_ratio1, npower=npower)).to(device)
        data_T = cast(Any, augment_batch(batch, aug2, aug_ratio2, npower=npower)).to(device)
        return data_S, data_T

    @torch.no_grad()
    def evaluate(self, model, loader, hypergraph_hie_aware_dict, device, anchor_queue=None, epoch=None):
        model.eval()

        total_correct = 0
        total_loss_sup = 0
        total_loss_hse = 0
        total_loss_con = 0.0
        num_samples = len(loader.loader.dataset)
        num_batches = len(loader)

        compute_loss_con = (
            anchor_queue is not None
            and len(anchor_queue) == self.configs["num_anchors"]
        )
        if compute_loss_con:
            S_anchors = torch.stack([x[0] for x in anchor_queue])
            T_anchors = torch.stack([x[1] for x in anchor_queue])

        loader.new_epoch()
        for _ in range(num_batches):
            batch = loader.next().to(device)
            H_S = self._batch_hypergraph(batch, hypergraph_hie_aware_dict, device)
            data_S, data_T = self._augment_views(batch, device, augment=False)
            S, T, out, loss_hse_S, loss_hse_T = model(data_S, data_T, H_S)

            total_loss_sup += float(F.cross_entropy(out, batch.y)) * batch.num_graphs
            total_loss_hse += float(
                self.configs["weight_hse"] * (loss_hse_S + loss_hse_T)) * batch.num_graphs
            if compute_loss_con:
                loss_con = model.loss_con(S, S_anchors, T, T_anchors)
                total_loss_con += float(self.configs["beta"] * loss_con)
            pred = out.softmax(dim=-1).argmax(dim=-1)
            total_correct += int((pred == batch.y).sum())

        return (
            total_correct / num_samples,
            total_loss_sup / num_samples,
            total_loss_hse / num_samples,
            total_loss_con / num_batches if compute_loss_con else 0.0,
        )

    def _create_model(self, dataset: TUDatasetExt, avg_num_nodes: int, device: torch.device) -> HypSEE:
        cfg = self.configs
        model = HypSEE(
            in_channels=dataset.num_features,
            hidden_channels_gnn=cfg["dim_embedding_gnn"],
            hidden_channels=cfg["dim_embedding"],
            out_channels=dataset.num_classes,
            num_layers_gnn=cfg["num_layers_gnn"],
            num_edges2=cfg["num_edges2"],
            avg_num_nodes=avg_num_nodes,
            height=cfg["height"],
            EPS=cfg["EPS"],
            decay_rate=cfg["decay_rate"],
            gnn_arch=cfg.get("gnn_arch", cfg.get("hgsl_arch", "GCN")),
            hgsl_constraint=cfg.get("hgsl_constraint", "sigmoid"),
            hgsl_topk=cfg.get("hgsl_topk") or None,
            use_gnn_encoder_S=cfg.get("use_gnn_encoder_S", False),
        )
        model.reset_parameters()
        return model.to(device)

    def _load_hypergraphs(self, dataset, seed=0):
        return load_hypergraphs(
            data_name=self.configs['data_name'],
            data_root=self.configs['data_root'],
            mode=self.configs['mode'],
            hyperedge_length_list=self.configs['hypergraph_length_list'],
            num_edges=self.configs['num_edges1'],
            num_graphs=len(dataset),
            seed=seed,
        )

    def _init_anchor_queue(self, model, labeled_loader, hypergraph_hie_aware_dict, device):
        num_anchors = self.configs["num_anchors"]
        with torch.no_grad():
            model.eval()
            labeled_loader.new_epoch()
            anchor_queue = []
            while len(anchor_queue) < num_anchors:
                labeled_batch = labeled_loader.next().to(device)
                labeled_H_S = self._batch_hypergraph(labeled_batch, hypergraph_hie_aware_dict, device)
                labeled_data_S, labeled_data_T = self._augment_views(labeled_batch, device)
                S_labeled, T_labeled, _, _, _ = model(labeled_data_S, labeled_data_T, labeled_H_S)
                need = num_anchors - len(anchor_queue)
                for i in range(min(S_labeled.shape[0], need)):
                    anchor_queue.append((S_labeled[i], T_labeled[i]))
            labeled_loader.new_epoch()
        assert len(anchor_queue) == num_anchors
        model.train()
        return anchor_queue

    def run(self, seed, device, exp_iter=0):
        self.fix_seed(seed)
        self._set_debug_ctx(seed=seed, epoch=None, phase="setup", model=None)

        dataset: TUDatasetExt = get_dataset(
            name=self.configs["data_name"],
            root=self.configs["data_root"],
            feat_str=self.configs["feat_str"],
        )
        dataset = shuffle_dataset(dataset)

        data = dataset._data
        if data is None or data.x is None:
            raise RuntimeError("dataset node features are not initialized")
        avg_num_nodes = int(data.x.size(0) / len(dataset))

        labeled_loader = IterLoader(DataLoader(
            subset_dataset(dataset, slice(None, 0.1)),
            batch_size=int(np.ceil(self.configs["batch_size"] / 5)),
            shuffle=False,
        ))
        unlabeled_loader = IterLoader(DataLoader(
            subset_dataset(dataset, slice(0.2, 0.7)),
            batch_size=self.configs["batch_size"],
            shuffle=False,
        ))
        val_loader = IterLoader(DataLoader(
            subset_dataset(dataset, slice(0.7, 0.8)),
            batch_size=self.configs["batch_size"],
            shuffle=False,
        ))
        test_loader = IterLoader(DataLoader(
            subset_dataset(dataset, slice(0.8, 1.0)),
            batch_size=self.configs["batch_size"],
            shuffle=False,
        ))

        model = self._create_model(dataset, avg_num_nodes, device)
        self._set_debug_ctx(model=model)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.configs["lr"],
            weight_decay=self.configs["weight_decay"],
        )

        best_epoch = None
        best_state = None
        best_val_loss_sup = 1e10
        best_val_loss_sup_se = 1e10
        best_val_acc = 0.0

        hypergraph_hie_aware_dict = None
        anchor_queue = []

        for epoch in range(1, self.configs["epochs"] + 1):
            self._set_debug_ctx(epoch=epoch)
            if self.configs['H1_update'] == 'epoch' or epoch == 1:
                self._set_debug_ctx(phase="load_hypergraphs")
                hypergraph_hie_aware_dict = self._load_hypergraphs(dataset, seed=seed + epoch)

            if epoch == 1 or self.configs['H1_update'] == 'epoch':
                assert hypergraph_hie_aware_dict is not None
                self._set_debug_ctx(phase="init_anchor_queue")
                anchor_queue = self._init_anchor_queue(
                    model, labeled_loader, hypergraph_hie_aware_dict, device)

            assert hypergraph_hie_aware_dict is not None
            self._set_debug_ctx(phase="train")
            train_loss, train_loss_sup, train_loss_con, train_loss_se = self.train(model, labeled_loader,
                                                                              unlabeled_loader, hypergraph_hie_aware_dict,
                                                                              optimizer, anchor_queue, epoch, device)
            self._set_debug_ctx(phase="evaluate")
            val_acc, val_loss_sup, val_loss_se, val_loss_con = self.evaluate(
                model, val_loader, hypergraph_hie_aware_dict, device,
                anchor_queue=anchor_queue, epoch=epoch)
            test_acc, test_loss_sup, test_loss_hse, test_loss_con = self.evaluate(
                model, test_loader, hypergraph_hie_aware_dict, device,
                anchor_queue=anchor_queue, epoch=epoch)
            print(
                "epoch{}:\ttrain loss: {}\ttrain_sup loss: {}\ttrain_con loss: {}\ttrain_hse loss: {}\t"
                "val_acc: {}\ttest_acc: {}"
                .format(epoch, train_loss, train_loss_sup, train_loss_con, train_loss_se, val_acc, test_acc))
            self._log_wandb_train(
                epoch, exp_iter, train_loss, train_loss_sup, train_loss_con, train_loss_se)
            self._log_wandb_eval(
                epoch, exp_iter, val_acc, val_loss_sup, val_loss_se, val_loss_con)
            self._log_wandb_test_epoch(
                epoch, exp_iter, test_acc, test_loss_sup, test_loss_hse, test_loss_con)
            if epoch > self.configs["warm_epochs"]:
                epoch_select = self.configs["epoch_select"]
                improved = False
                score = 0.0
                if epoch_select == 'val_loss_sup':
                    if best_state is None or val_loss_sup < best_val_loss_sup:
                        best_val_loss_sup = val_loss_sup
                        improved = True
                        score = val_loss_sup
                elif epoch_select == 'val_acc':
                    if best_state is None or val_acc > best_val_acc:
                        best_val_acc = val_acc
                        improved = True
                        score = val_acc
                elif epoch_select == 'val_acc_eq':
                    if best_state is None or val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        improved = True
                        score = val_acc
                elif epoch_select == 'val_loss_sup_hse':
                    val_loss_sup_hse = val_loss_sup + val_loss_se
                    if best_state is None or val_loss_sup_hse < best_val_loss_sup_se:
                        best_val_loss_sup_se = val_loss_sup_hse
                        improved = True
                        score = val_loss_sup_hse
                else:
                    raise NotImplementedError(epoch_select)

                if improved:
                    best_epoch = epoch
                    best_state = deepcopy(model.state_dict())
                    print("epoch{}:\t{}={}".format(best_epoch, epoch_select, score))

        if best_state is not None:
            model.load_state_dict(best_state)
        test_acc, test_loss_sup, test_loss_hse, test_loss_con = self.evaluate(
            model, test_loader, hypergraph_hie_aware_dict, device,
            anchor_queue=anchor_queue, epoch=self.configs["epochs"])
        print(
            "best_epoch:{}\ttest_acc:{}\ttest_loss_sup:{}\ttest_loss_con:{}\ttest_loss_hse:{}"
            .format(best_epoch, test_acc, test_loss_sup, test_loss_con, test_loss_hse))
        return test_acc, best_epoch, test_loss_sup, test_loss_hse, test_loss_con



    def exp(self):
        print(self.configs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.configs["data_name"], self.configs["data_root"])

        if "hypergraph_length_list" not in self.configs:
            self.configs["hypergraph_length_list"] = [4, 6]
        if self.configs["data_name"] == "COLLAB" and self.configs["feat_str"] == "":
            self.configs["feat_str"] = "deg+odeg100"

        test_acc_list = []
        seed = 0
        exp_iter = 0
        while len(test_acc_list) < self.configs["runs"]:
            test_acc, best_epoch, test_loss_sup, test_loss_hse, test_loss_con = self.run(
                seed, device, exp_iter=exp_iter)
            print(f"run: {len(test_acc_list)}\tacc: {test_acc}")
            test_acc_list.append(test_acc)
            self._log_iter_metrics(
                exp_iter, test_acc, test_loss_sup, test_loss_hse, test_loss_con,
                best_epoch, seed, test_acc_list)
            exp_iter += 1
            seed += 1

        mean_acc = float(np.mean(test_acc_list))
        std_acc = float(np.std(test_acc_list))
        print(f"test_acc_mean: {mean_acc}\tstd: {std_acc}")
        if self.wandb is not None:
            self.wandb.log({
                'exp_iter': exp_iter,
                'final/avg_test_acc': mean_acc,
                'final/std_test_acc': std_acc,
            })
            self.wandb.summary.update({
                'final/avg_test_acc': mean_acc,
                'final/std_test_acc': std_acc,
            })
        return mean_acc
