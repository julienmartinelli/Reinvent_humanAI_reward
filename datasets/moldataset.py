import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .basedataset import BaseDataset


class MolDataHuman(BaseDataset):
    """DRD2 dataset class."""

    def __init__(
        self,
        train_features,
        train_labels,
        test_features,
        test_labels,
        val_split=0.1,
        batch_size=100,
        transforms=None,
        seed=0,
    ):
        """
        n_train_samples: total number of train samples in the dataset
        n_test_samples: total number of test samples in the dataset
        """
        torch.manual_seed(seed)
        self.val_split = val_split
        self.batch_size = batch_size
        self.transforms = transforms
        self.train_features = torch.tensor(train_features)
        self.train_labels = torch.tensor(train_labels)
        self.test_features = torch.tensor(test_features)
        self.test_labels = torch.tensor(test_labels)
        self.n_train_samples = train_features.shape[0]
        self.n_test_samples = test_features.shape[0]
        self.n_samples = train_features.shape[0] + test_features.shape[0]
        self.d = train_features.shape[1]
        self.generate_data()

    def generate_data(self):

        # stratified split of train set to train and validation sets
        train_idx, validation_idx = train_test_split(
            np.arange(self.n_train_samples),
            test_size=self.val_split,
            random_state=42,
            shuffle=True,
            stratify=self.train_labels,
        )

        sub_train_features, sub_train_labels = torch.utils.data.Subset(
            self.train_features, train_idx
        ), torch.utils.data.Subset(self.train_labels, train_idx)
        val_features, val_labels = torch.utils.data.Subset(
            self.train_features, validation_idx
        ), torch.utils.data.Subset(self.train_labels, validation_idx)

        opt_rej_preds_sub_train = self.get_rej_preds(
            sub_train_features.dataset.data.float()
        )
        opt_rej_preds_val = self.get_rej_preds(val_features.dataset.data.float())
        opt_rej_preds_test = self.get_rej_preds(self.test_features.data.float())

        human_predictions_sub_train = self.get_human_predictions(
            sub_train_labels.dataset.data, opt_rej_preds_sub_train
        )  # where human predictions is equal to real labels
        human_predictions_val = self.get_human_predictions(
            val_labels.dataset.data, opt_rej_preds_val
        )
        human_predictions_test = self.get_human_predictions(
            self.test_labels, opt_rej_preds_test
        )

        print("train size: ", len(sub_train_features))
        print("val size: ", len(val_features))
        print("test size: ", len(self.test_features))

        self.data_train = torch.utils.data.TensorDataset(
            sub_train_features.dataset.data.float(),
            sub_train_labels.dataset.data,
            human_predictions_sub_train,
        )
        self.data_val = torch.utils.data.TensorDataset(
            val_features.dataset.data.float(),
            val_labels.dataset.data,
            human_predictions_val,
        )
        self.data_test = torch.utils.data.TensorDataset(
            self.test_features.float(), self.test_labels, human_predictions_test
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=True
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=True
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            self.data_test, batch_size=self.batch_size, shuffle=True
        )

        # At the beginning, when there has not been any human query yet, D_h is simply D_l without the y labels

        self.data_train_dh = torch.utils.data.TensorDataset(
            sub_train_features.dataset.data.float(), human_predictions_sub_train
        )
        self.data_val_dh = torch.utils.data.TensorDataset(
            val_features.dataset.data.float(), human_predictions_val
        )
        self.data_test_dh = torch.utils.data.TensorDataset(
            self.test_features.float(), human_predictions_test
        )

        self.data_train_dh_loader = torch.utils.data.DataLoader(
            self.data_train_dh, batch_size=self.batch_size, shuffle=True
        )
        self.data_val_dh_loader = torch.utils.data.DataLoader(
            self.data_val_dh, batch_size=self.batch_size, shuffle=True
        )
        self.data_test_dh_loader = torch.utils.data.DataLoader(
            self.data_test_dh, batch_size=self.batch_size, shuffle=True
        )

    def update_dh(self, x, h):
        new_queries = torch.utils.data.TensorDataset(
            torch.from_numpy(x, device=self.device),
            torch.from_numpy(h, device=self.device),
        )
        self.data_train_dh = torch.utils.data.ConcatDataset(
            self.data_train_dh, new_queries
        )
        self.data_train_dh_loader = torch.utils.data.DataLoader(
            self.data_train_dh, batch_size=self.batch_size, shuffle=True
        )
