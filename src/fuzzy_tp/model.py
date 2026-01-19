from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add, scatter_mean


TNorm = Literal["product", "lukasiewicz", "godel"]
SupportAgg = Literal["prob_sum", "lukasiewicz", "godel"]
RNNType = Literal["gru", "lstm"]
GatingMode = Literal["atom", "edge"]
AtomGateCount = Literal[2, 3]


class SigmoidSpreadNormalizer(nn.Module):
    """Normalizes probabilities by scaling logits before applying a sigmoid."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        # Store log(alpha) so spreads remain positive after exponentiation.
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, probs: Tensor) -> Tensor:
        if probs.numel() == 0:
            return probs
        clamped = torch.clamp(probs, self.eps, 1.0 - self.eps)
        logits = torch.logit(clamped)
        alpha = torch.exp(self.log_alpha).to(dtype=logits.dtype, device=logits.device)
        scaled = alpha * logits
        return torch.sigmoid(scaled)


class StableModelGNN(nn.Module):
    """Graph neural network that approximates fuzzy stable models over RH graphs."""

    def __init__(
        self,
        *,
        num_iters: int = 50,
        tnorm: TNorm = "product",
        support_aggregation: SupportAgg = "prob_sum",
        gating_mode: GatingMode = "atom",
        atom_gate_count: AtomGateCount = 2,
        rnn_type: RNNType = "gru",
        atom_hidden_dim: int = 64,
        atom_input_dim: int = 7,
        logical_dimension: int = 1,
        use_gnn_backbone: bool = False,
        use_wfs_bounds: bool = False,
        use_convergence: bool = True,
        convergence_tol: float = 1e-4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_iters < 1:
            raise ValueError("num_iters must be at least 1")
        if atom_hidden_dim < 0:
            raise ValueError("atom_hidden_dim must be non-negative")
        if convergence_tol <= 0.0:
            raise ValueError("convergence_tol must be positive")
        self.num_iters = num_iters
        self.max_iters = num_iters
        self.tnorm: TNorm = tnorm
        self.support_aggregation: SupportAgg = support_aggregation
        if gating_mode not in ("atom", "edge"):
            raise ValueError(f"Unsupported gating_mode '{gating_mode}'")
        self.gating_mode: GatingMode = gating_mode
        if atom_gate_count not in (2, 3):
            raise ValueError(f"atom_gate_count must be 2 or 3, got {atom_gate_count}")
        self.atom_gate_count: AtomGateCount = atom_gate_count
        self.rnn_type: RNNType = rnn_type
        self.eps = eps
        if logical_dimension < 1:
            raise ValueError("logical_dimension must be at least 1")
        self.logical_dimension = logical_dimension
        self.atom_hidden_dim = atom_hidden_dim
        self.use_gnn_backbone = use_gnn_backbone and atom_hidden_dim > 0
        self.use_convergence = use_convergence
        self.convergence_tol = convergence_tol
        self.atom_input_dim = atom_input_dim
        self.use_wfs_bounds = use_wfs_bounds
        # if self.atom_input_dim < 4:
        #     raise ValueError("atom_input_dim must be at least 4 to accommodate mean delta feature.")
        self.layer_normalizers = nn.ModuleList(
            SigmoidSpreadNormalizer(eps=self.eps) for _ in range(self.max_iters + 1)
        )

        if atom_hidden_dim > 0:
            rnn_input_size = self.logical_dimension * self.atom_input_dim
            if use_gnn_backbone and atom_hidden_dim > 0:
                rnn_input_size += self.logical_dimension * atom_hidden_dim
            if rnn_type == "gru":
                self.atom_rnn: nn.Module = nn.GRU(
                    input_size=rnn_input_size,
                    hidden_size=atom_hidden_dim,
                    batch_first=False,
                )
            elif rnn_type == "lstm":
                self.atom_rnn = nn.LSTM(
                    input_size=rnn_input_size,
                    hidden_size=atom_hidden_dim,
                    batch_first=False,
                )
            else:
                raise ValueError(f"Unsupported rnn_type '{rnn_type}'")
            self.atom_update = nn.Linear(atom_hidden_dim, atom_gate_count)
        else:
            self.atom_rnn = None
            self.atom_update = None

        if self.gating_mode == "edge":
            edge_input_dim = 2
            hidden_gate = max(atom_hidden_dim, 4)
            self.edge_gate: nn.Module = nn.Sequential(
                nn.Linear(edge_input_dim, hidden_gate),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_gate, 1),
            )
        else:
            self.edge_gate = None

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _ensure_layer_normalizers(self, required: int) -> None:
        while len(self.layer_normalizers) < required:
            self.layer_normalizers.append(SigmoidSpreadNormalizer(eps=self.eps))

    def _apply_layer_normalizer(self, probs: Tensor, layer_idx: int) -> Tensor:
        if probs.numel() == 0:
            return probs
        self._ensure_layer_normalizers(layer_idx + 1)
        normalizer = self.layer_normalizers[layer_idx]
        return normalizer(probs)

    def _negate(self, values: Tensor) -> Tensor:
        return 1.0 - values

    def _aggregate_literals(self, values: Tensor, rule_index: Tensor, dim_size: int) -> Tensor:
        if values.numel() == 0:
            return values.new_ones(dim_size, self.logical_dimension)
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        if self.tnorm == "product":
            clamped = values.clamp(self.eps, 1.0 - self.eps)
            log_values = torch.log(clamped)
            acc = scatter_add(log_values, rule_index, dim=0, dim_size=dim_size)
            return torch.exp(acc)
        if self.tnorm == "lukasiewicz":
            sums = scatter_add(values, rule_index, dim=0, dim_size=dim_size)
            counts = scatter_add(torch.ones_like(values), rule_index, dim=0, dim_size=dim_size)
            combined = sums - (counts - 1.0)
            return combined  # .clamp(0.0, 1.0)
        if self.tnorm == "godel":
            minima = torch.ones(dim_size, self.logical_dimension, device=values.device, dtype=values.dtype)
            if values.numel() > 0:
                rule_index_expanded = rule_index.unsqueeze(-1).expand(-1, self.logical_dimension)
                minima = minima.scatter_reduce(
                    0,
                    rule_index_expanded,
                    values,
                    reduce="amin",
                    include_self=True,
                )
            return minima  # .clamp(0.0, 1.0)
        raise ValueError(f"Unsupported tnorm '{self.tnorm}'")

    def _combine_tnorm(self, a: Tensor, b: Tensor) -> Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        if b.dim() == 1:
            b = b.unsqueeze(-1)
        if self.tnorm == "product":
            return (a * b)  # .clamp(0.0, 1.0)
        if self.tnorm == "lukasiewicz":
            return torch.clamp(a + b - 1.0, 0.0, 1.0)
        if self.tnorm == "godel":
            return torch.minimum(a, b)
        raise ValueError(f"Unsupported tnorm '{self.tnorm}'")

    def _rule_body_value(
        self,
        atom_probs: Tensor,
        edge_pos: Tensor,
        edge_neg: Tensor,
        num_rules: int,
        negation_override: Optional[Tensor] = None,
    ) -> Tensor:
        if num_rules == 0:
            return atom_probs.new_empty(0, self.logical_dimension)

        device = atom_probs.device
        dtype = atom_probs.dtype
        body_value = torch.ones(num_rules, self.logical_dimension, device=device, dtype=dtype)

        pos_vals: Optional[Tensor] = None
        neg_vals: Optional[Tensor] = None

        if edge_pos.numel() > 0:
            pos_vals = atom_probs[edge_pos[0]]
            pos_tnorm = self._aggregate_literals(pos_vals, edge_pos[1], num_rules)
            body_value = self._combine_tnorm(body_value, pos_tnorm)

        if edge_neg.numel() > 0:
            if negation_override is not None:
                neg_vals = negation_override.to(device=atom_probs.device, dtype=atom_probs.dtype)[edge_neg[0]]
            else:
                neg_vals = self._negate(atom_probs[edge_neg[0]])
            neg_tnorm = self._aggregate_literals(neg_vals, edge_neg[1], num_rules)
            body_value = self._combine_tnorm(body_value, neg_tnorm)

        return body_value

    def _evaluate_rules(
        self,
        atom_probs: Tensor,
        edge_pos: Tensor,
        edge_neg: Tensor,
        num_rules: int,
        negation_override: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        body_value = self._rule_body_value(
            atom_probs,
            edge_pos,
            edge_neg,
            num_rules,
            negation_override,
        )
        empty_features = body_value.new_empty((body_value.size(0), 0))
        return body_value, body_value, empty_features

    def _aggregate_support(
        self,
        rule_support: Tensor,
        head_edge: Tensor,
        num_atoms: int,
        edge_messages: Optional[Tensor] = None,
    ) -> Tensor:
        if num_atoms == 0 or rule_support.numel() == 0 or head_edge.numel() == 0:
            return torch.zeros(num_atoms, self.logical_dimension, device=rule_support.device, dtype=rule_support.dtype)

        if rule_support.dim() == 1:
            rule_support = rule_support.unsqueeze(-1)
        if edge_messages is not None and edge_messages.dim() == 1:
            edge_messages = edge_messages.unsqueeze(-1)

        rule_idx = head_edge[0]
        atom_idx = head_edge[1]
        messages = edge_messages if edge_messages is not None else rule_support[rule_idx]

        ones = torch.ones(messages.size(0), self.logical_dimension, device=messages.device, dtype=messages.dtype)
        counts = scatter_add(ones, atom_idx, dim=0, dim_size=num_atoms)

        if self.support_aggregation == "godel":
            support = torch.zeros(num_atoms, self.logical_dimension, device=messages.device, dtype=messages.dtype)
            if messages.numel() > 0:
                base = torch.full_like(support, fill_value=-float("inf"))
                atom_idx_expanded = atom_idx.unsqueeze(-1).expand(-1, self.logical_dimension)
                base = base.scatter_reduce(
                    0,
                    atom_idx_expanded,
                    messages,
                    reduce="amax",
                    include_self=True,
                )
                mask = base > -float("inf")
                support = torch.where(mask, base, support)
            return support
        if self.support_aggregation == "prob_sum":
            one_minus = (1.0 - messages).clamp(self.eps, 1.0)
            log_acc = scatter_add(torch.log(one_minus), atom_idx, dim=0, dim_size=num_atoms)
            support = (1.0 - torch.exp(log_acc)).clamp(0.0, 1.0)
            support[counts == 0] = 0.0
            return support
        if self.support_aggregation == "lukasiewicz":
            summed = scatter_add(messages, atom_idx, dim=0, dim_size=num_atoms)
            support = summed.clamp(0.0, 1.0)
            support[counts == 0] = 0.0
            return support
        raise ValueError(f"Unsupported support aggregation '{self.support_aggregation}'")

    def _prepare_fact_overrides(
        self,
        num_rules: int,
        fact_mask: Optional[Tensor],
        fact_values: Optional[Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if fact_mask is None:
            if fact_values is not None:
                raise ValueError("fact_values provided but fact_mask is None.")
            return None, None

        mask_tensor = fact_mask.to(device=device, dtype=torch.bool).view(-1)
        if mask_tensor.numel() < num_rules:
            raise ValueError("fact_mask has fewer elements than num_rules.")
        mask_tensor = mask_tensor[:num_rules]

        if mask_tensor.numel() == 0 or not mask_tensor.any():
            return mask_tensor, None if fact_values is None else torch.zeros(num_rules, self.logical_dimension, device=device, dtype=dtype)

        if fact_values is None:
            overrides = torch.ones(num_rules, self.logical_dimension, device=device, dtype=dtype)
            return mask_tensor, overrides

        value_tensor = fact_values.to(device=device, dtype=dtype)
        if value_tensor.dim() == 1:
            value_tensor = value_tensor.unsqueeze(-1)
        elif value_tensor.dim() != 2:
            raise ValueError("fact_values must be a 1D or 2D tensor.")

        if value_tensor.size(-1) == 1 and self.logical_dimension > 1:
            value_tensor = value_tensor.expand(-1, self.logical_dimension)
        elif value_tensor.size(-1) != self.logical_dimension:
            raise ValueError("fact_values second dimension does not match logical_dimension.")

        if value_tensor.size(0) == num_rules:
            overrides = value_tensor
        elif value_tensor.size(0) == int(mask_tensor.sum().item()):
            overrides = torch.ones(num_rules, self.logical_dimension, device=device, dtype=dtype)
            overrides[mask_tensor] = value_tensor
        else:
            raise ValueError("fact_values length must match num_rules or number of facts.")

        return mask_tensor, overrides

    # Public helpers -------------------------------------------------
    def compute_rule_support(
        self,
        atom_probs: Tensor,
        edge_pos: Tensor,
        edge_neg: Tensor,
        num_rules: int,
        negation_override: Optional[Tensor] = None,
        *,
        fact_mask: Optional[Tensor] = None,
        fact_values: Optional[Tensor] = None,
        keepdim: bool = False,
    ) -> Tensor:
        """Expose rule-body evaluation for analysis utilities."""
        support = self._rule_body_value(
            atom_probs,
            edge_pos,
            edge_neg,
            num_rules,
            negation_override,
        )
        if fact_mask is not None or fact_values is not None:
            mask_tensor, overrides = self._prepare_fact_overrides(
                num_rules,
                fact_mask,
                fact_values,
                device=support.device,
                dtype=support.dtype,
            )
            if mask_tensor is not None and mask_tensor.any():
                replacement = overrides
                if replacement is None:
                    replacement = torch.ones(
                        num_rules,
                        self.logical_dimension,
                        device=support.device,
                        dtype=support.dtype,
                    )
                support = support.clone()
                support[mask_tensor] = replacement[mask_tensor]
        if not keepdim and self.logical_dimension == 1:
            return support.squeeze(-1)
        return support

    def aggregate_rule_support(
        self,
        rule_support: Tensor,
        head_edge: Tensor,
        num_atoms: int,
        edge_messages: Optional[Tensor] = None,
        *,
        keepdim: bool = False,
    ) -> Tensor:
        """Aggregate rule support into atom-level support via the configured t-conorm."""
        support = self._aggregate_support(rule_support, head_edge, num_atoms, edge_messages=edge_messages)
        if not keepdim and self.logical_dimension == 1:
            return support.squeeze(-1)
        return support

    def _graph_mean_feature(self, values: Tensor, atom_ptr: Optional[Tensor]) -> Tensor:
        if values.numel() == 0:
            return values.clone()
        squeezed = False
        if values.dim() == 1:
            values = values.unsqueeze(-1)
            squeezed = True
        if atom_ptr is None or atom_ptr.numel() <= 1:
            mean_val = values.mean(dim=0, keepdim=True)
            expanded = mean_val.expand(values.size(0), -1)
            return expanded.squeeze(-1) if squeezed else expanded
        ptr = atom_ptr.to(device=values.device, dtype=torch.long)
        if ptr.numel() < 2:
            zeros = torch.zeros_like(values)
            return zeros.squeeze(-1) if squeezed else zeros
        lengths = ptr[1:] - ptr[:-1]
        if lengths.numel() == 0 or int(lengths.sum().item()) == 0:
            zeros = torch.zeros_like(values)
            return zeros.squeeze(-1) if squeezed else zeros
        total = int(lengths.sum().item())
        if total != values.size(0):
            mean_val = values.mean(dim=0, keepdim=True)
            expanded = mean_val.expand(values.size(0), -1)
            return expanded.squeeze(-1) if squeezed else expanded
        segment_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=values.device, dtype=torch.long),
            lengths,
            output_size=values.size(0),
        )
        means = scatter_mean(values, segment_ids, dim=0, dim_size=lengths.numel())
        result = means[segment_ids]
        return result.squeeze(-1) if squeezed else result

    def _graph_min_abs_half(self, values: Tensor, atom_ptr: Optional[Tensor]) -> Tensor:
        """Per-graph minimum distance to 0.5, expanded to atom shape."""
        if values.numel() == 0:
            return torch.zeros_like(values)
        squeezed = False
        if values.dim() == 1:
            values = values.unsqueeze(-1)
            squeezed = True
        abs_diff = torch.abs(values - 0.5)
        if atom_ptr is None or atom_ptr.numel() <= 1:
            min_val = abs_diff.min(dim=0, keepdim=True).values
            expanded = min_val.expand(values.size(0), -1)
            return expanded.squeeze(-1) if squeezed else expanded
        ptr = atom_ptr.to(device=values.device, dtype=torch.long)
        if ptr.numel() < 2:
            zeros = torch.zeros_like(values)
            return zeros.squeeze(-1) if squeezed else zeros
        lengths = ptr[1:] - ptr[:-1]
        if lengths.numel() == 0 or int(lengths.sum().item()) == 0:
            zeros = torch.zeros_like(values)
            return zeros.squeeze(-1) if squeezed else zeros
        total = int(lengths.sum().item())
        if total != values.size(0):
            min_val = abs_diff.min(dim=0, keepdim=True).values
            expanded = min_val.expand(values.size(0), -1)
            return expanded.squeeze(-1) if squeezed else expanded
        segment_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=values.device, dtype=torch.long),
            lengths,
            output_size=values.size(0),
        )
        mins = torch.full((lengths.numel(), abs_diff.size(-1)), float("inf"), device=values.device, dtype=values.dtype)
        mins = mins.scatter_reduce(
            0,
            segment_ids.unsqueeze(-1).expand(-1, abs_diff.size(-1)),
            abs_diff,
            reduce="amin",
            include_self=True,
        )
        mins_expanded = mins[segment_ids]
        return mins_expanded.squeeze(-1) if squeezed else mins_expanded

    def graph_mean_feature(self, values: Tensor, atom_ptr: Optional[Tensor]) -> Tensor:
        """Public wrapper around graph-level mean feature computation."""
        return self._graph_mean_feature(values, atom_ptr)

    def _wfs_refine_bounds(
        self,
        lower: Tensor,
        upper: Tensor,
        edge_pos: Tensor,
        edge_neg: Tensor,
        head_edge: Tensor,
        num_rules: int,
    ) -> Tuple[Tensor, Tensor]:
        """One deterministic well-founded-style refinement step on (lower, upper)."""
        if lower.numel() == 0 or upper.numel() == 0 or num_rules == 0:
            return lower, upper

        # True/false assumptions for negation when computing the optimistic/pessimistic closures.
        false_for_lower = 1.0 - upper
        false_for_upper = 1.0 - lower

        lower_rule_support = self.compute_rule_support(
            lower,
            edge_pos,
            edge_neg,
            num_rules,
            negation_override=false_for_lower,
            keepdim=True,
        )
        lower_atom_support = self.aggregate_rule_support(
            lower_rule_support,
            head_edge,
            lower.size(0),
            keepdim=True,
        )

        upper_rule_support = self.compute_rule_support(
            upper,
            edge_pos,
            edge_neg,
            num_rules,
            negation_override=false_for_upper,
            keepdim=True,
        )
        upper_atom_support = self.aggregate_rule_support(
            upper_rule_support,
            head_edge,
            upper.size(0),
            keepdim=True,
        )

        refined_lower = torch.maximum(lower, lower_atom_support)
        refined_upper = torch.minimum(upper, upper_atom_support)
        return refined_lower, refined_upper

    def _init_hidden(
        self,
        num_atoms: int,
        device: torch.device,
    ) -> Optional[Tuple[Tensor, ...]]:
        if self.atom_rnn is None or num_atoms == 0:
            return None
        hidden_dim = self.atom_rnn.hidden_size
        param_dtype = next(self.atom_rnn.parameters()).dtype
        units = num_atoms * self.logical_dimension
        hidden = torch.rand(1, units, hidden_dim, device=device, dtype=param_dtype)
        if self.rnn_type == "lstm":
            cell = torch.rand(1, units, hidden_dim, device=device, dtype=param_dtype)
            return hidden, cell
        return (hidden,)

    def initialize_hidden(
        self,
        num_atoms: int,
        device: torch.device,
    ) -> Optional[Tuple[Tensor, ...]]:
        """Public helper that exposes hidden-state initialisation."""
        return self._init_hidden(num_atoms, device)

    def _gnn_backbone_messages(
        self,
        atom_hidden: Tensor,
        edge_pos: Tensor,
        edge_neg: Tensor,
        head_edge: Tensor,
        num_atoms: int,
        num_rules: int,
        *,
        add_self: bool = True,
    ) -> Tensor:
        if (not self.use_gnn_backbone) or num_atoms == 0:
            return atom_hidden.new_zeros((num_atoms, self.logical_dimension, atom_hidden.size(-1)))

        hidden_dim = atom_hidden.size(-1)
        device = atom_hidden.device
        dtype = atom_hidden.dtype

        if num_rules == 0:
            return atom_hidden.clone()

        rule_messages = torch.zeros(num_rules, self.logical_dimension, hidden_dim, device=device, dtype=dtype)
        rule_counts = torch.zeros(num_rules, device=device, dtype=dtype)

        if edge_pos.numel() > 0:
            pos_atoms = atom_hidden[edge_pos[0]]
            rule_messages.index_add_(0, edge_pos[1], pos_atoms)
            rule_counts.index_add_(0, edge_pos[1], torch.ones(edge_pos.size(1), device=device, dtype=dtype))

        if edge_neg.numel() > 0:
            neg_atoms = atom_hidden[edge_neg[0]]
            rule_messages.index_add_(0, edge_neg[1], neg_atoms)
            rule_counts.index_add_(0, edge_neg[1], torch.ones(edge_neg.size(1), device=device, dtype=dtype))

        valid_rules = rule_counts > 0
        if valid_rules.any():
            rule_messages[valid_rules] = rule_messages[valid_rules] / rule_counts[valid_rules].view(-1, 1, 1)

        atom_messages = torch.zeros(num_atoms, self.logical_dimension, hidden_dim, device=device, dtype=dtype)
        atom_counts = torch.zeros(num_atoms, device=device, dtype=dtype)

        if head_edge.numel() > 0 and rule_messages.numel() > 0:
            atom_messages.index_add_(0, head_edge[1], rule_messages[head_edge[0]])
            atom_counts.index_add_(0, head_edge[1], torch.ones(head_edge.size(1), device=device, dtype=dtype))

        valid_atoms = atom_counts > 0
        if valid_atoms.any():
            atom_messages[valid_atoms] = atom_messages[valid_atoms] / atom_counts[valid_atoms].view(-1, 1, 1)

        if add_self:
            atom_messages += atom_hidden
        return atom_messages

    def gnn_backbone_messages(
        self,
        atom_hidden: Tensor,
        edge_pos: Tensor,
        edge_neg: Tensor,
        head_edge: Tensor,
        num_atoms: int,
        num_rules: int,
        *,
        add_self: bool = True,
    ) -> Tensor:
        """Public helper to compute optional backbone messages."""
        return self._gnn_backbone_messages(
            atom_hidden,
            edge_pos,
            edge_neg,
            head_edge,
            num_atoms,
            num_rules,
            add_self=add_self,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        p0: Tensor,
        edge_pos: Tensor,
        edge_neg: Tensor,
        head_edge: Tensor,
        num_atoms: int,
        num_rules: int,
        *,
        atom_ptr: Optional[Tensor] = None,
        rule_ptr: Optional[Tensor] = None,
        fact_mask: Optional[Tensor] = None,
        fact_values: Optional[Tensor] = None,
        return_iterates: bool = False,
        return_debug: bool = False,
        max_iters: Optional[int] = None,
        use_convergence: Optional[bool] = None,
        convergence_tol: Optional[float] = None,
        negation_override: Optional[Tensor] = None,
        init_function = None,
        return_stability: bool = False,
    ) -> Tensor | Tuple[Tensor, list[Tensor]]:
        if num_atoms == 0:
            empty_base = p0.new_empty(0)
            if return_iterates:
                return empty_base, []
            return empty_base

        if p0.size(0) != num_atoms:
            raise ValueError(f"num_atoms ({num_atoms}) does not match p0 size ({p0.size(0)})")

        if p0.dim() == 1:
            expanded = p0.unsqueeze(-1).expand(num_atoms, self.logical_dimension)
            probs = expanded.clone()
        elif p0.dim() == 2:
            if p0.size(1) != self.logical_dimension:
                raise ValueError(
                    f"p0 second dimension ({p0.size(1)}) does not match logical_dimension ({self.logical_dimension})",
                )
            probs = p0.clone()
        else:
            raise ValueError("p0 must be a 1D or 2D tensor")

        if probs.size(0) != num_atoms:
            raise ValueError(
                f"num_atoms ({num_atoms}) does not match p0 size ({probs.size(0)})",
            )

        if fact_values is not None and fact_mask is None:
            raise ValueError("fact_values provided but fact_mask is None.")

        fact_rule_mask: Optional[Tensor] = None
        fact_rule_values: Optional[Tensor] = None
        if fact_mask is not None:
            fact_rule_mask, fact_rule_values = self._prepare_fact_overrides(
                num_rules,
                fact_mask,
                fact_values,
                device=probs.device,
                dtype=probs.dtype,
            )

        fact_atom_indicator: Optional[Tensor] = None
        if (
            fact_rule_mask is not None
            and fact_rule_mask.numel() > 0
            and fact_rule_mask.any()
            and head_edge.numel() > 0
            and num_rules > 0
        ):
            fact_values_for_rules = fact_rule_values
            if fact_values_for_rules is None:
                fact_values_for_rules = torch.ones(
                    num_rules,
                    self.logical_dimension,
                    device=probs.device,
                    dtype=probs.dtype,
                )
            if head_edge.dim() != 2:
                raise ValueError("head_edge must be a 2 x E tensor.")
            head_edge_for_facts = head_edge if head_edge.device == probs.device else head_edge.to(device=probs.device)
            rule_indices = head_edge_for_facts[0]
            fact_edge_mask = fact_rule_mask[rule_indices]
            if fact_edge_mask.any():
                head_edge_facts = head_edge_for_facts[:, fact_edge_mask]
                rule_indices_facts = head_edge_facts[0]
                edge_messages = fact_values_for_rules[rule_indices_facts]
                fact_atom_indicator = self._aggregate_support(
                    fact_values_for_rules,
                    head_edge_facts,
                    num_atoms,
                    edge_messages=edge_messages,
                )
            else:
                fact_atom_indicator = torch.zeros(
                    num_atoms,
                    self.logical_dimension,
                    device=probs.device,
                    dtype=probs.dtype,
                )

        atom_ptr_device = atom_ptr.to(device=probs.device) if atom_ptr is not None else None
        rule_ptr_device = rule_ptr.to(device=probs.device) if rule_ptr is not None else None
        if init_function is None:
            hidden = self._init_hidden(num_atoms, probs.device)
        else:
            hidden = init_function(num_atoms, probs.device)

        global_lower = probs.new_zeros(num_atoms, self.logical_dimension)
        global_upper = probs.new_ones(num_atoms, self.logical_dimension)

        collect_iterates = return_iterates or return_debug
        iterates: list[Tensor] = []
        if collect_iterates:
            iterates.append(probs.clone())

        support_history: list[Tensor] = []
        rule_support_history: list[Tensor] = []
        gate_history: list[Tensor] = []
        hidden_history: list[Tensor] = []
        cell_history: list[Tensor] = []
        gnn_history: list[Tensor] = []
        edge_alpha_history: list[Tensor] = []

        limit = max_iters if max_iters is not None else self.max_iters
        if limit < 1:
            raise ValueError("max_iters must be at least 1")
        conv_enabled = self.use_convergence if use_convergence is None else use_convergence
        tol = self.convergence_tol if convergence_tol is None else convergence_tol
        if conv_enabled and tol <= 0.0:
            raise ValueError("convergence_tol must be positive")
        self._ensure_layer_normalizers(limit + 1)
        layer_index = 0
        bounds_fixed = False

        for _ in range(limit):
            prev_probs = probs.clone() if conv_enabled else None
            if self.use_wfs_bounds and not bounds_fixed:
                refined_lower, refined_upper = self._wfs_refine_bounds(
                    global_lower,
                    global_upper,
                    edge_pos,
                    edge_neg,
                    head_edge,
                    num_rules,
                )
                if torch.equal(refined_lower, global_lower) and torch.equal(refined_upper, global_upper):
                    bounds_fixed = True
                else:
                    global_lower, global_upper = refined_lower, refined_upper

            rule_support = self.compute_rule_support(
                probs,
                edge_pos,
                edge_neg,
                num_rules,
                negation_override,
                fact_mask=fact_rule_mask,
                fact_values=fact_rule_values,
                keepdim=True,
            )
            if return_debug:
                rule_support_history.append(rule_support.clone())

            edge_messages: Optional[Tensor] = None
            if self.gating_mode == "edge":
                if head_edge.numel() > 0:
                    rule_indices = head_edge[0]
                    atom_indices = head_edge[1]
                    edge_features = torch.stack(
                        [
                            rule_support[rule_indices],
                            probs[atom_indices],
                        ],
                        dim=-1,
                    )
                    flat_edge_features = edge_features.view(-1, edge_features.size(-1))
                    flat_alpha = torch.sigmoid(self.edge_gate(flat_edge_features))
                    edge_alpha = flat_alpha.view(edge_features.size(0), self.logical_dimension)
                    edge_messages = rule_support[rule_indices] * edge_alpha
                    if return_debug:
                        edge_alpha_history.append(edge_alpha.clone())
                elif return_debug:
                    edge_alpha_history.append(rule_support.new_zeros((0, self.logical_dimension)))

            atom_hidden_state: Optional[Tensor] = None
            if hidden is not None:
                hidden_tensor = hidden[0] if self.rnn_type == "gru" else hidden[0]
                atom_hidden_state = hidden_tensor.squeeze(0).view(num_atoms, self.logical_dimension, -1)

            atom_support = self.aggregate_rule_support(
                rule_support,
                head_edge,
                num_atoms,
                edge_messages=edge_messages,
                keepdim=True,
            )
            if return_debug:
                support_history.append(atom_support.clone())
            delta = atom_support - probs
            mean_feature = self._graph_mean_feature(delta, atom_ptr_device)
            local_lower = torch.minimum(probs, atom_support)
            local_upper = torch.maximum(probs, atom_support)
            # ℓ_P(I^t) ∩ U^{t+1},  u_P(I^t) ∪ L^{t+1}
            combined_lower = torch.minimum(local_lower, global_upper)
            combined_upper = torch.maximum(local_upper, global_lower)
            closest_to_half = self._graph_min_abs_half(probs, atom_ptr_device)
            base_features = torch.stack(
                [atom_support, probs, delta, mean_feature, combined_lower, combined_upper, closest_to_half],
                dim=-1,
            )
            relaxed_lower = combined_lower
            if self.use_gnn_backbone and atom_hidden_state is not None:
                gnn_messages = self._gnn_backbone_messages(
                    atom_hidden_state,
                    edge_pos,
                    edge_neg,
                    head_edge,
                    num_atoms,
                    num_rules,
                )
                if return_debug:
                    gnn_history.append(gnn_messages.clone())
            else:
                gnn_messages = None

            if self.atom_rnn is not None:
                base_flat = base_features.reshape(num_atoms, self.logical_dimension * base_features.size(-1))
                if gnn_messages is not None:
                    gnn_flat = gnn_messages.reshape(num_atoms, self.logical_dimension * gnn_messages.size(-1))
                    per_atom_input = torch.cat([base_flat, gnn_flat], dim=-1)
                else:
                    per_atom_input = base_flat
                expanded_input = per_atom_input.unsqueeze(1).expand(-1, self.logical_dimension, -1)
                flat_atom_input = expanded_input.reshape(num_atoms * self.logical_dimension, -1)
                rnn_in = flat_atom_input.unsqueeze(0)
                if self.rnn_type == "gru":
                    if hidden is None:
                        raise RuntimeError("GRU hidden state is None despite atom_rnn being defined.")
                    if not isinstance(hidden, tuple) or hidden[0] is None:
                        raise RuntimeError("GRU hidden tuple is malformed.")
                    rnn_out, gru_hidden = self.atom_rnn(rnn_in, hidden[0])
                    hidden = (gru_hidden,)
                    current_hidden = gru_hidden
                    current_cell = None
                else:
                    if hidden is None:
                        raise RuntimeError("LSTM hidden state is None despite atom_rnn being defined.")
                    rnn_out, lstm_state = self.atom_rnn(rnn_in, hidden)
                    hidden = lstm_state
                    current_hidden, current_cell = lstm_state
                rnn_out = rnn_out.squeeze(0).view(num_atoms, self.logical_dimension, -1)
                flat_rnn_out = rnn_out.view(num_atoms * self.logical_dimension, -1)
                if return_debug:
                    hidden_snapshot = current_hidden.squeeze(0).view(num_atoms, self.logical_dimension, -1).clone()
                    hidden_history.append(hidden_snapshot)
                    if self.rnn_type == "lstm":
                        if current_cell is None:
                            raise RuntimeError("Expected current_cell for LSTM but found None.")
                        cell_snapshot = current_cell.squeeze(0).view(num_atoms, self.logical_dimension, -1).clone()
                        cell_history.append(cell_snapshot)
            else:
                rnn_out = None
                if return_debug and self.atom_hidden_dim > 0:
                    hidden_history.append(
                        torch.zeros(
                            num_atoms,
                            self.logical_dimension,
                            self.atom_hidden_dim,
                            device=probs.device,
                        ),
                    )

            gate_weights: Optional[Tensor]
            if self.gating_mode == "atom":
                if self.atom_rnn is not None:
                    logits = self.atom_update(flat_rnn_out)
                    gate_weights = torch.softmax(logits, dim=-1).view(
                        num_atoms,
                        self.logical_dimension,
                        self.atom_gate_count,
                    )
                    combined = gate_weights[..., 0] * relaxed_lower + gate_weights[..., 1] * atom_support
                    if self.atom_gate_count == 3:
                        fact_multiplier = (
                            fact_atom_indicator
                            if fact_atom_indicator is not None
                            else torch.zeros(
                                num_atoms,
                                self.logical_dimension,
                                device=probs.device,
                                dtype=probs.dtype,
                            )
                        )
                        combined = combined + gate_weights[..., 2] * fact_multiplier
                    probs = combined
                    probs = probs.clamp_min(0.0)
                else:
                    gate_weights = torch.zeros(
                        probs.size(0),
                        self.logical_dimension,
                        self.atom_gate_count,
                        device=probs.device,
                        dtype=probs.dtype,
                    )
                    gate_weights[..., 1] = 1.0
                    probs = gate_weights[..., 0] * relaxed_lower + gate_weights[..., 1] * atom_support
                    if self.atom_gate_count == 3:
                        fact_multiplier = (
                            fact_atom_indicator
                            if fact_atom_indicator is not None
                            else torch.zeros(
                                num_atoms,
                                self.logical_dimension,
                                device=probs.device,
                                dtype=probs.dtype,
                            )
                        )
                        probs = probs + gate_weights[..., 2] * fact_multiplier
            else:
                gate_weights = None
                probs = atom_support

            probs = torch.minimum(torch.maximum(probs, relaxed_lower), combined_upper)
            if return_debug and gate_weights is not None:
                gate_history.append(gate_weights.clone())

            probs = self._apply_layer_normalizer(probs, layer_index)
            layer_index += 1

            if collect_iterates:
                iterates.append(probs.clone())

            if conv_enabled and prev_probs is not None:
                max_delta = torch.max(torch.abs(probs - prev_probs)).item()
                if max_delta <= tol:
                    break

        pre_final_probs = probs.clone()

        rule_support = self.compute_rule_support(
            probs,
            edge_pos,
            edge_neg,
            num_rules,
            negation_override,
            fact_mask=fact_rule_mask,
            fact_values=fact_rule_values,
            keepdim=True,
        )
        if return_debug:
            rule_support_history.append(rule_support.clone())

        if self.gating_mode == "edge":
            if head_edge.numel() > 0:
                rule_indices = head_edge[0]
                atom_indices = head_edge[1]
                edge_features_final = torch.stack(
                    [
                        rule_support[rule_indices],
                        probs[atom_indices],
                    ],
                    dim=-1,
                )
                flat_edge_features_final = edge_features_final.view(-1, edge_features_final.size(-1))
                flat_alpha_final = torch.sigmoid(self.edge_gate(flat_edge_features_final))
                edge_alpha_final = flat_alpha_final.view(edge_features_final.size(0), self.logical_dimension)
                edge_messages_final = rule_support[rule_indices] * edge_alpha_final
            else:
                edge_alpha_final = rule_support.new_zeros((0, self.logical_dimension))
                edge_messages_final = None
            atom_support = self._aggregate_support(
                rule_support,
                head_edge,
                num_atoms,
                edge_messages=edge_messages_final,
            )
            if return_debug:
                edge_alpha_history.append(edge_alpha_final.clone())
        elif num_atoms > 0 and head_edge.numel() > 0 and num_rules > 0:
            atom_support = self._aggregate_support(rule_support, head_edge, num_atoms)
        else:
            atom_support = probs
        probs = atom_support # torch.clamp(atom_support, 0.0, 1.0)
        if return_debug:
            support_history.append(atom_support.clone())
            if self.gating_mode == "atom":
                final_gate = probs.new_zeros(probs.size(0), self.logical_dimension, self.atom_gate_count)
                if self.atom_gate_count > 1:
                    final_gate[..., 1] = 1.0
                gate_history.append(final_gate)
            if self.atom_rnn is not None:
                final_hidden = hidden[0].squeeze(0).view(num_atoms, self.logical_dimension, -1).clone()
                hidden_history.append(final_hidden)
                if self.rnn_type == "lstm":
                    final_cell = hidden[1].squeeze(0).view(num_atoms, self.logical_dimension, -1).clone()
                    cell_history.append(final_cell)
            elif self.atom_hidden_dim > 0:
                hidden_history.append(
                    torch.zeros(
                        probs.size(0),
                        self.logical_dimension,
                        self.atom_hidden_dim,
                        device=probs.device,
                    ),
                )

        probs = self._apply_layer_normalizer(probs, layer_index)
        layer_index += 1

        final_delta = torch.abs(probs - pre_final_probs)
        binary_gap = torch.minimum(probs, 1.0 - probs)

        if atom_ptr_device is not None and atom_ptr_device.numel() > 1:
            atom_ptr_long = atom_ptr_device.to(device=probs.device, dtype=torch.long)
            atom_segments = [
                (
                    int(atom_ptr_long[i].item()),
                    int(atom_ptr_long[i + 1].item()),
                )
                for i in range(atom_ptr_long.numel() - 1)
            ]
        else:
            atom_ptr_long = None
            atom_segments = [(0, num_atoms)]

        num_segments = len(atom_segments)
        selected_dims_tensor = torch.zeros(max(1, num_segments), dtype=torch.long, device=probs.device)

        for seg_idx, (start, end) in enumerate(atom_segments):
            length = max(0, end - start)
            if length <= 0:
                selected_dims_tensor[seg_idx] = 0
                continue

            seg_delta = final_delta[start:end]
            seg_binary_gap = binary_gap[start:end]
            stability_mean = seg_delta.mean(dim=0)
            binary_mean = seg_binary_gap.mean(dim=0)
            binary_max = seg_binary_gap.max(dim=0).values

            best_dim = 0
            best_stability = stability_mean[0]
            best_binary = binary_mean[0]
            best_is_stable = best_stability <= tol
            best_max_gap = binary_max[0]
            best_is_bad = best_max_gap <= 0.01

            for dim in range(1, self.logical_dimension):
                dim_stability = stability_mean[dim]
                dim_binary = binary_mean[dim]
                dim_max_gap = binary_max[dim]
                dim_is_stable = dim_stability <= tol
                dim_is_bad = dim_max_gap <= 0.01

                if dim_is_bad:
                    if (
                        not best_is_bad
                        or dim_max_gap < best_max_gap - self.eps
                        or (abs(dim_max_gap - best_max_gap) <= self.eps and dim_binary + self.eps < best_binary)
                    ):
                        best_dim = dim
                        best_stability = dim_stability
                        best_binary = dim_binary
                        best_is_stable = dim_is_stable
                        best_max_gap = dim_max_gap
                        best_is_bad = True
                elif best_is_bad:
                    continue
                elif dim_is_stable:
                    if (
                        not best_is_stable
                        or dim_binary + self.eps < best_binary
                        or (abs(dim_binary - best_binary) <= self.eps and dim_stability + self.eps < best_stability)
                    ):
                        best_dim = dim
                        best_stability = dim_stability
                        best_binary = dim_binary
                        best_is_stable = True
                        best_max_gap = dim_max_gap
                        best_is_bad = False
                else:
                    if not best_is_stable:
                        if (
                            dim_stability + self.eps < best_stability
                            or (abs(dim_stability - best_stability) <= self.eps and dim_binary + self.eps < best_binary)
                        ):
                            best_dim = dim
                            best_stability = dim_stability
                            best_binary = dim_binary
                            best_max_gap = dim_max_gap
                            best_is_stable = False
                            best_is_bad = False

            selected_dims_tensor[seg_idx] = best_dim
        # print(probs[selected_dims_tensor], pre_final_probs[selected_dims_tensor])
        segment_lengths = torch.tensor(
            [max(0, end - start) for start, end in atom_segments],
            dtype=torch.long,
            device=probs.device,
        )
        if segment_lengths.numel() > 0 and int(segment_lengths.sum().item()) > 0:
            atom_dim_indices = torch.repeat_interleave(
                selected_dims_tensor[:segment_lengths.numel()],
                segment_lengths,
                dim=0,
            )
        else:
            atom_dim_indices = torch.empty(0, dtype=torch.long, device=probs.device)

        if atom_dim_indices.numel() < num_atoms:
            pad_dim = selected_dims_tensor[-1:].expand(num_atoms - atom_dim_indices.numel())
            atom_dim_indices = torch.cat([atom_dim_indices, pad_dim], dim=0)
        elif atom_dim_indices.numel() > num_atoms:
            atom_dim_indices = atom_dim_indices[:num_atoms]

        if selected_dims_tensor is None or atom_dim_indices is None:
            selected_dims_tensor = probs.new_zeros(1, dtype=torch.long, device=probs.device)
            atom_dim_indices = selected_dims_tensor.expand(num_atoms)
        else:
            selected_dims_tensor = selected_dims_tensor.to(device=probs.device)
            atom_dim_indices = atom_dim_indices.to(device=probs.device)

        selected_output = torch.gather(probs, 1, atom_dim_indices.unsqueeze(-1)).squeeze(-1)
        selected_dims = selected_dims_tensor.detach().cpu().tolist()

        if rule_ptr_device is not None and rule_ptr_device.numel() > 1:
            rule_segments = [
                (
                    int(rule_ptr_device[i].item()),
                    int(rule_ptr_device[i + 1].item()),
                )
                for i in range(rule_ptr_device.numel() - 1)
            ]
        else:
            rule_segments = [(0, num_rules)]
        if not rule_segments:
            rule_segments = [(0, num_rules)]

        rule_dim_lookup = probs.new_zeros(num_rules, dtype=torch.long)
        if num_rules > 0:
            for seg_idx, (start, end) in enumerate(rule_segments):
                dim_idx = selected_dims[min(seg_idx, len(selected_dims) - 1)]
                start_idx = max(0, start)
                end_idx = min(num_rules, max(start_idx, end))
                if end_idx > start_idx:
                    rule_dim_lookup[start_idx:end_idx] = dim_idx

        num_head_edges = head_edge.size(1)
        edge_rule_indices = head_edge[0] if head_edge.numel() > 0 else None

        def project_atom_tensor(tensor: Tensor) -> Tensor:
            if tensor.dim() == 0 or tensor.numel() == 0:
                return tensor
            if tensor.dim() == 1 or tensor.size(0) != num_atoms:
                return tensor
            if tensor.size(1) != self.logical_dimension:
                raise RuntimeError("Atom-aligned tensor has unexpected logical dimension.")
            trailing = tensor.shape[2:]
            projected = tensor.new_empty((num_atoms,) + trailing)
            for seg_idx, (start, end) in enumerate(atom_segments):
                dim_idx = selected_dims[min(seg_idx, len(selected_dims) - 1)]
                if end <= start:
                    continue
                projected[start:end] = tensor[start:end, dim_idx]
            return projected

        def project_rule_tensor(tensor: Tensor) -> Tensor:
            if tensor.dim() == 0 or tensor.numel() == 0:
                return tensor
            if tensor.dim() == 1 or tensor.size(0) != num_rules:
                return tensor
            if tensor.size(1) != self.logical_dimension:
                raise RuntimeError("Rule-aligned tensor has unexpected logical dimension.")
            trailing = tensor.shape[2:]
            projected = tensor.new_empty((num_rules,) + trailing)
            if num_rules == 0:
                return projected
            for rule_idx in range(num_rules):
                dim_idx = int(
                    rule_dim_lookup[rule_idx].item()
                    if rule_dim_lookup.numel() > rule_idx
                    else selected_dims[0]
                )
                projected[rule_idx] = tensor[rule_idx, dim_idx]
            return projected

        def project_edge_tensor(tensor: Tensor) -> Tensor:
            if tensor.dim() == 0 or tensor.numel() == 0 or num_head_edges == 0:
                return tensor
            if tensor.dim() == 1 or tensor.size(0) != num_head_edges:
                return tensor
            if tensor.size(1) != self.logical_dimension:
                raise RuntimeError("Edge-aligned tensor has unexpected logical dimension.")
            projected = tensor.new_empty(num_head_edges)
            if edge_rule_indices is None:
                return projected
            for edge_idx in range(num_head_edges):
                rule_idx = int(edge_rule_indices[edge_idx].item())
                dim_idx = int(
                    rule_dim_lookup[rule_idx].item()
                    if 0 <= rule_idx < rule_dim_lookup.numel()
                    else selected_dims[0]
                )
                projected[edge_idx] = tensor[edge_idx, dim_idx]
            return projected

        iterates_projected = [project_atom_tensor(tensor) for tensor in iterates] if iterates else []
        support_projected = [project_atom_tensor(tensor) for tensor in support_history]
        rule_support_projected = [project_rule_tensor(tensor) for tensor in rule_support_history]
        gate_projected = [project_atom_tensor(tensor) for tensor in gate_history]
        hidden_projected = [project_atom_tensor(tensor) for tensor in hidden_history]
        cell_projected = [project_atom_tensor(tensor) for tensor in cell_history]
        gnn_projected = [project_atom_tensor(tensor) for tensor in gnn_history]
        edge_alpha_projected = [project_edge_tensor(tensor) for tensor in edge_alpha_history]
        stability_projected = project_atom_tensor(final_delta)

        if return_debug:
            debug: dict[str, torch.Tensor] = {}
            if iterates:
                debug["probabilities_all"] = torch.stack([tensor.detach().cpu() for tensor in iterates], dim=0)
            if iterates_projected:
                debug["probabilities"] = torch.stack(
                    [tensor.detach().cpu() for tensor in iterates_projected],
                    dim=0,
                )
            if support_projected:
                debug["supports"] = torch.stack([tensor.detach().cpu() for tensor in support_projected], dim=0)
            if rule_support_projected:
                debug["rule_supports"] = torch.stack(
                    [tensor.detach().cpu() for tensor in rule_support_projected],
                    dim=0,
                )
            if gate_projected:
                debug["gates"] = torch.stack([tensor.detach().cpu() for tensor in gate_projected], dim=0)
            if hidden_projected and self.atom_hidden_dim > 0:
                debug["hidden"] = torch.stack([tensor.detach().cpu() for tensor in hidden_projected], dim=0)
            if (
                cell_projected
                and self.rnn_type == "lstm"
                and self.atom_hidden_dim > 0
            ):
                debug["cell"] = torch.stack([tensor.detach().cpu() for tensor in cell_projected], dim=0)
            if gnn_projected:
                debug["gnn_messages"] = torch.stack(
                    [tensor.detach().cpu() for tensor in gnn_projected],
                    dim=0,
                )
            if edge_alpha_projected:
                debug["edge_alpha"] = torch.stack(
                    [tensor.detach().cpu() for tensor in edge_alpha_projected],
                    dim=0,
                )
            debug["stability_delta"] = stability_projected.detach().cpu()
            debug["selected_dimensions"] = selected_dims_tensor.detach().cpu()
            if selected_dims_tensor.numel() == 1:
                debug["selected_dimension"] = selected_dims_tensor.detach().cpu()[0]

            return selected_output, debug

        if return_iterates:
            if return_stability:
                return selected_output, iterates_projected, stability_projected
            return selected_output, iterates_projected
        if return_stability:
            return selected_output, stability_projected
        return selected_output
