import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, Optional
from .blocks import Exp_Scale
from .losses import sigmoid_focal_loss
from metric import topk_accuracy


class PrefixPromptTextEncoder:
    """Encodes action prompts with a learnable prefix through the frozen CLIP text tower (Eq. 7).

        y_a = E(p || Embed(Tokenize(P_a)))

    The prefix p (M token embeddings, shared by all actions) is inserted right after the
    start-of-text token, the sequence is shifted right by M (dropping M trailing padding
    tokens so the context length stays fixed), and the frozen text transformer is run
    exactly as open_clip's ``TextTransformer.forward`` does: positional embedding, causal
    mask, final LayerNorm, pooling at the (shifted) end-of-text token, text projection.
    Gradients reach only the prefix; the CLIP weights stay frozen.
    """

    @staticmethod
    def encode(clip_model, text_tokens: torch.Tensor, prefix: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            clip_model: (clip, tokenizer, preprocess) tuple; clip must be an open_clip model
                with a separate text tower (e.g. EVA02-L-14-336, a CustomTextCLIP).
            text_tokens: (C, L) token ids of the C action prompts.
            prefix: (M, D) learnable prefix embeddings, or None for the plain frozen encoder.

        Returns:
            (C, embed_dim) text features.
        """
        clip = clip_model[0]
        if prefix is None:
            return clip.encode_text(text_tokens)

        text = clip.text
        if text.cls_emb is not None or text.pool_type != 'argmax':
            raise NotImplementedError("prefix prompts assume argmax (EOT) pooling and no CLS embedding")

        num_prompts, context_len = text_tokens.shape
        prefix_len = prefix.shape[0]
        eot_idx = text_tokens.argmax(dim=-1)  # EOT has the largest token id
        assert int(eot_idx.max()) + prefix_len < context_len, \
            "prompt + prefix exceeds the CLIP context length"
        assert prefix.shape[1] == text.token_embedding.embedding_dim, \
            "prefix dim must match the CLIP token-embedding width"

        cast_dtype = text.transformer.get_cast_dtype()
        token_embeds = text.token_embedding(text_tokens).to(cast_dtype)          # (C, L, D)
        prefix = prefix.to(cast_dtype).unsqueeze(0).expand(num_prompts, -1, -1)  # (C, M, D)
        x = torch.cat([token_embeds[:, :1], prefix, token_embeds[:, 1:context_len - prefix_len]], dim=1)

        x = x + text.positional_embedding.to(cast_dtype)
        x = text.transformer(x, attn_mask=text.attn_mask)
        x = text.ln_final(x)
        pooled = x[torch.arange(num_prompts, device=x.device), eot_idx + prefix_len]

        if text.text_projection is not None:
            if isinstance(text.text_projection, nn.Linear):
                pooled = text.text_projection(pooled)
            else:
                pooled = pooled @ text.text_projection
        return pooled


class ActionSimilarityComputer:
    """Computes similarity between visual and text features."""

    def __init__(self, logits_scale: nn.Module, cos_sim: nn.Module):
        self.logits_scale = logits_scale
        self.cos_sim = cos_sim

    def compute_training_logits(self, vis_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        """Compute logits for training with temperature scaling.

        Args:
            vis_feat: Visual feature tensor.
            txt_feat: Text feature tensor.

        Returns:
            Scaled logits tensor of shape (batch_size, num_classes).
        """
        vis_feat_norm = vis_feat / torch.norm(vis_feat, dim=-1, keepdim=True)
        txt_feat_norm = txt_feat / torch.norm(txt_feat, dim=-1, keepdim=True)

        logits = self.logits_scale(vis_feat_norm @ txt_feat_norm.t())
        return logits.reshape(-1, txt_feat.shape[0])

    def compute_inference_similarity(self, vis_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        """Compute similarity for inference using cosine similarity.

        Args:
            vis_feat: Visual feature tensor.
            txt_feat: Text feature tensor.

        Returns:
            Normalized similarity scores in range [0, 1].
        """
        score = self.cos_sim(vis_feat, txt_feat)
        return (score + 1) / 2  # Normalize to [0, 1]


class Action_Model(nn.Module):
    """
    Prompt-based mistake detector.

    Each action segment embedding is aligned with the prompt of its action, encoded by the
    frozen CLIP text tower with a learnable prefix (Eq. 7). Training combines
      - the softmax contrastive loss of Eq. 8 over the C action prompts of the task, and
      - a sigmoid focal loss on the same logits, i.e. a per-prompt binary
        "matches / does not match" objective.
    At inference the cosine similarity to the predicted action's prompt is the normality
    score (mistake probability = 1 - similarity).
    """

    def __init__(self, vis_input_dim: int, txt_input_dim: int, feat_dim: int,
                 num_actions: int, prefix_length: int = 4,
                 ce_loss_weight: float = 1.0, focal_loss_weight: float = 1.0):
        super().__init__()

        # Store dimensions
        self.vis_input_dim = vis_input_dim
        self.txt_input_dim = txt_input_dim
        self.num_actions = num_actions
        self.feat_dim = feat_dim
        self.ce_loss_weight = ce_loss_weight
        self.focal_loss_weight = focal_loss_weight

        # Initialize model components
        self._init_learnable_parameters(prefix_length, txt_input_dim, vis_input_dim, feat_dim)
        self._init_similarity_components()

        # Initialize helper classes
        self.similarity_computer = ActionSimilarityComputer(self.logits_scale, self.cos_sim)

    def _init_learnable_parameters(self, prefix_length: int, txt_input_dim: int,
                                  vis_input_dim: int, feat_dim: int):
        """Initialize the prompt prefix and the projections into the joint space."""
        # Learnable prompt prefix p of Eq. 7, in the CLIP token-embedding space; std 0.02
        # matches the scale of CLIP's own token embeddings. prefix_length = 0 disables it
        # (fixed prompts encoded by the frozen text encoder only).
        if prefix_length > 0:
            self.prompt_prefix = nn.Parameter(torch.empty(prefix_length, txt_input_dim))
            nn.init.normal_(self.prompt_prefix, std=0.02)
        else:
            self.prompt_prefix = None

        # Projection matrices
        vis_scale = vis_input_dim ** -0.5
        txt_scale = txt_input_dim ** -0.5

        self.vis_proj = nn.Parameter(vis_scale * torch.randn(vis_input_dim, feat_dim))
        self.txt_proj = nn.Parameter(txt_scale * torch.randn(txt_input_dim, feat_dim))

    def _init_similarity_components(self):
        """Initialize components for similarity computation."""
        self.logits_scale = Exp_Scale(np.log(1 / 0.07))
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, clip_model, segment_vis_feature: torch.Tensor, action_tokens: torch.Tensor,
                action_labels: List[torch.Tensor], device: str = "cuda") -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Forward pass for training.

        Args:
            clip_model: CLIP model instance for text and visual encoding.
            segment_vis_feature: Visual features for action segments.
            action_tokens: Text tokens representing action classes.
            action_labels: List of ground truth action label tensors.
            device: Target device for computation (default: "cuda").

        Returns:
            Tuple of the detection losses ('det_ce_loss' and/or 'det_focal_loss', already
            weighted, summed over segments) and a dictionary of top-k accuracy metrics.
        """
        # Encode text and visual features
        encoded_features = self._encode_features(
            clip_model, segment_vis_feature, action_tokens
        )

        # Prepare labels
        labels = self._prepare_labels(action_labels, device)

        # Compute logits: cos(x_a, y_c) / rho for every segment and every action prompt
        logits = self.similarity_computer.compute_training_logits(
            encoded_features['visual'], encoded_features['text']
        )

        # Summed over segments like the other auxiliary losses; the meta-architecture divides
        # all of them by its running number of positive points.
        losses = {}
        if self.ce_loss_weight > 0:
            # Eq. 8: softmax contrastive loss over the C action prompts of the task
            losses['det_ce_loss'] = self.ce_loss_weight * F.cross_entropy(
                logits, labels['class_ids'], reduction='sum')
        if self.focal_loss_weight > 0:
            losses['det_focal_loss'] = self.focal_loss_weight * sigmoid_focal_loss(
                logits, labels['onehot'], reduction='sum')

        # Compute accuracy metrics
        topk_acc = topk_accuracy(
            labels['class_ids'].clone(), logits.clone(),
            topk=(1, 3, 5), name="action"
        )

        return losses, topk_acc

    @torch.no_grad()
    def inference(self, clip_model, segment_vis_feature: torch.Tensor, action_tokens: torch.Tensor,
                  action_labels: List[torch.Tensor], device: str = "cuda") -> torch.Tensor:
        """Inference pass for computing action similarities.

        Args:
            clip_model: CLIP model instance for text and visual encoding.
            segment_vis_feature: Visual features for action segments.
            action_tokens: Text tokens representing action classes.
            action_labels: List of action label tensors for segments.
            device: Target device for computation (default: "cuda").

        Returns:
            Similarity scores tensor between visual features and corresponding text features.
        """
        # Encode features
        encoded_features = self._encode_features(
            clip_model, segment_vis_feature, action_tokens
        )

        # Get segment-specific text features
        clip_action_labels = torch.cat(action_labels, dim=0).to(device)
        segment_text_feat = encoded_features['text'][clip_action_labels]

        # Compute similarity
        return self.similarity_computer.compute_inference_similarity(
            encoded_features['visual'], segment_text_feat
        )

    def _encode_features(self, clip_model, segment_vis_feature: torch.Tensor,
                        action_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode visual and text features through projection layers.

        Args:
            clip_model: CLIP model instance for text encoding.
            segment_vis_feature: Visual features for segments.
            action_tokens: Text tokens for actions.

        Returns:
            Dictionary containing 'visual' and 'text' projected feature tensors.
        """
        text_features = PrefixPromptTextEncoder.encode(clip_model, action_tokens, self.prompt_prefix)
        text_features = text_features.to(self.txt_proj.dtype) @ self.txt_proj
        visual_features = segment_vis_feature @ self.vis_proj

        return {
            'visual': visual_features,
            'text': text_features
        }

    def _prepare_labels(self, action_labels: List[torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        """Prepare labels for loss computation.

        Args:
            action_labels: List of action label tensors from different segments.
            device: Target device for tensors.

        Returns:
            Dictionary containing 'class_ids' (merged label tensor) and 'onehot' (one-hot encoded labels).
        """
        # Merge labels across batches
        clip_action_labels = torch.cat(action_labels, dim=0).to(device)

        # Create one-hot encoding
        onehot_labels = F.one_hot(clip_action_labels, num_classes=self.num_actions).float()

        return {
            'class_ids': clip_action_labels,
            'onehot': onehot_labels
        }
