import numpy as np
import torch
import torch.nn.functional as F


class QuerySelector():
    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader
        assert dataloader.dataset.queries == None
        self.queries = {}
        self.device = args.device
        self.uc_sampler = UncertaintySampler(args.query_strategy)
        self.query_strategy = args.query_strategy
        self.n_pixels_per_round = args.n_pixels_per_round
        self.top_n_percent = args.top_n_percent

    def gen_init_queries(self):
        assert len(self.queries) == 0
        for imgs, labels, img_names in self.dataloader:
            img, label, img_name = imgs[0], labels[0], img_names[0]
            self.queries[img_name] = []
            class_ids = np.unique(label)
            label_flat = label.flatten()
            for class_id in class_ids:
                query_pool = np.where(label_flat == class_id)[0]
                self.queries[img_name].extend(np.random.choice(
                    query_pool, min(self.args.n_init_pixels_per_class, len(query_pool)), replace=False))
        return self.queries

    @torch.no_grad()
    def __call__(self, model):
        for imgs, labels, img_names in self.dataloader:
            # get uncertainty map
            inputs, labels = imgs.to(self.device), labels.long().to(self.device)
            outputs = model(inputs)
            prob = F.softmax(outputs, dim=1)
            uncertainty_map = self.uc_sampler(prob).squeeze(dim=0)
            # exclude queried pixels
            uc_map_flat = uncertainty_map.flatten()
            uc_map_flat[self.queries[img_names[0]]] = 0.0 if self.query_strategy in ["entropy", "least_confidence"] else 1.0
            # get top k pixels or random k pixels from top n%
            k = int(uc_map_flat.shape[0] * self.top_n_percent) if self.top_n_percent > 0. else self.n_pixels_per_round
            query_ind = uc_map_flat.topk(k=k, largest=self.query_strategy in ["entropy", "least_confidence"]).indices.cpu().numpy()
            if self.top_n_percent > 0.:
                query_ind = np.random.choice(query_ind, self.n_pixels_per_round, replace=False)
            self.queries[img_names[0]].extend(query_ind)
        return self.queries


class UncertaintySampler:
    def __init__(self, query_strategy):
        self.query_strategy = query_strategy

    @staticmethod
    def _entropy(prob):
        return (-prob * torch.log(prob)).sum(dim=1)  # b x h x w

    @staticmethod
    def _least_confidence(prob):
        return 1.0 - prob.max(dim=1)[0]  # b x h x w

    @staticmethod
    def _margin(prob):
        top2 = prob.topk(k=2, dim=1).values  # b x k x h x w
        return (top2[:, 0, :, :] - top2[:, 1, :, :]).abs()  # b x h x w

    @staticmethod
    def _random(prob):
        b, _, h, w = prob.shape
        return torch.rand((b, h, w))

    def __call__(self, prob):
        return getattr(self, f"_{self.query_strategy}")(prob)
