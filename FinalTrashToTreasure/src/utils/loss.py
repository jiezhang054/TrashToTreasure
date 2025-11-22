import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ReconstructionLoss, self).__init__()
        self.reduction = reduction

    def forward(self, reconstructed_list, original_list):
        total_loss = 0

        for x_hat, x_original in zip(reconstructed_list, original_list):
            # Calculate reconstruction error for each view
            mse_loss = F.mse_loss(x_hat, x_original, reduction=self.reduction)
            total_loss += mse_loss

        # Process according to reduction type
        if self.reduction == 'mean':
            total_loss = total_loss / len(original_list)
        elif self.reduction == 'sum':
            pass

        return total_loss


class UsefulLoss(nn.Module):
    def __init__(self, config, reduction='mean', class_weights=None):
        super(UsefulLoss, self).__init__()
        self.config = config
        self.num_classes = int(self.config.num_classes)
        self.latent_dim = int(self.config.latent_dim)
        self.num_views = self.config.num_views
        self.reduction = reduction
        self.class_weights = class_weights

        # Create learnable classifiers for each view
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_classes)
            ) for _ in range(self.num_views)
        ])

    def forward(self, useful_representations, labels):

        total_loss = 0
        view_losses = []

        for i, useful in enumerate(useful_representations):
            classifier = self.classifiers[i]
            logits = classifier(useful)

            if self.class_weights is not None:
                loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            else:
                loss = F.cross_entropy(logits, labels)

            view_losses.append(loss)
            total_loss += loss

        if self.reduction == 'mean':
            total_loss = sum(view_losses) / len(view_losses)
        elif self.reduction == 'sum':
            pass

        return total_loss


class GapLoss(nn.Module):
    def __init__(self, config, reduction='mean', class_weights=None):
        super(GapLoss, self).__init__()
        self.config = config
        self.num_views = self.config.num_views
        self.num_classes = int(self.config.num_classes)
        self.latent_dim = int(self.config.latent_dim)
        self.reduction = reduction
        self.class_weights = class_weights

        self.treasure_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )
        self.trash_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_classes)
            ) for _ in range(self.num_views)
        ])

    def forward(self, treasure_representation, trash_representations, labels):
        treasure_loss = self.estimate_mutual_info(treasure_representation, labels, self.treasure_classifier)

        trash_loss_list = []

        for i,trash in enumerate(trash_representations):
            trash_loss = self.estimate_mutual_info(trash, labels,self.trash_classifiers[i])
            trash_loss_list.append(trash_loss)

        avg_trash_loss = torch.stack(trash_loss_list).mean()
        gap_loss = treasure_loss - avg_trash_loss

        return gap_loss

    def estimate_mutual_info(self, representation, labels, classifiers):

        logits = classifiers(representation)

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights,reduction=self.reduction)
        else:
            loss = F.cross_entropy(logits, labels,reduction=self.reduction)

        return loss