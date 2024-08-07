import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss adapted for protein sequence embeddings.
    This version uses log_softmax for numerical stability and handles both 1D and 2D labels.
    """

    def __init__(self, temperature=0.1):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, output_1, output_2, labels):
        # Normalize the embeddings
        output_1 = F.normalize(output_1, dim=-1)
        output_2 = F.normalize(output_2, dim=-1)

        # Concatenate the embeddings from both views
        embeddings = torch.cat([output_1, output_2], dim=0)

        # Check if labels are 1D or 2D, and repeat accordingly
        if labels.dim() == 1:
            labels = labels.repeat(2)  # Repeat labels for both sets of embeddings
        elif labels.dim() == 2:
            labels = labels.repeat(2, 1)  # Repeat along the batch dimension

        # Calculate similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.transpose(-1, -2))

        # Create a mask for all positive samples
        labels = labels.unsqueeze(1)  # Prepare labels for broadcasting
        mask_positive = torch.eq(labels, labels.transpose(-1, -2))

        # Use log_softmax for numerical stability
        log_prob = F.log_softmax(similarity_matrix / self.temperature, dim=1)

        # Compute the loss for each sample
        loss = -torch.sum(log_prob * mask_positive, dim=1) / mask_positive.sum(dim=1)

        # Average the loss across all samples
        loss = loss.mean()

        return loss


class CombinedLoss(nn.Module):
    def __init__(self, temperature=0.1, contrastive_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.contrastive_loss = MultiPosConLoss(temperature)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_weight = (
            contrastive_weight  # weighting factor for the two losses
        )

    def forward(self, output_1, output_2, labels, logits):
        # Compute the binary cross-entropy loss
        bce_loss = self.bce_loss(logits, labels)

        if self.contrastive_weight != 0:
            # Compute the contrastive loss
            contrastive_loss = self.contrastive_loss(output_1, output_2, labels)

            # Combine the two losses
            combined_loss = (
                self.contrastive_weight * contrastive_loss
                + (1 - self.contrastive_weight) * bce_loss
            )
        else:
            combined_loss = bce_loss

        return combined_loss
