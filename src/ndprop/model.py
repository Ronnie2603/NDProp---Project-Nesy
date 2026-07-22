from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add

TNorm = Literal["product", "lukasiewicz", "godel"]
DecisionNormalization = Literal["none", "program"]


def _clamp01(x: Tensor) -> Tensor:
    return torch.clamp(x, 0.0, 1.0)


class NDProp(nn.Module):
    """Continuous relaxation of decision-based partial update semantics.

    Each atom keeps (tau, phi) in [0,1]. Propagation is deterministic: K add-only
    steps that relax the completion fixpoint with tau <- tau oplus u and
    phi <- phi oplus w (w aggregates failure of all rule bodies). A separate
    decision controller (RNN_dec) maintains state s^t across outer iterations,
    produces logits over atoms, and a masked sigmoid biases toward undecided
    atoms m = (1 - tau) otimes (1 - phi). Phi is increased by m odot pi; final
    belief is p = 0.5 * (tau + (1 - phi)).
    """

    def __init__(
        self,
        *,
        num_iters: int = 30,
        tnorm: TNorm = "product",
        tnorm_wfs: Optional[TNorm] = None,
        hidden_dim: int = 64,
        logical_dimension: int = 1,
        eps: float = 1e-6,
        propagation_steps: int = 10,
        use_convergence: bool = True,
        convergence_tol: float = 0,
        weak_negation_falsity: bool = False,
        decision_normalization: DecisionNormalization = "none",
    ) -> None:
        super().__init__()
        if num_iters < 1:
            raise ValueError("num_iters must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if logical_dimension < 1:
            raise ValueError("logical_dimension must be at least 1")
        if propagation_steps < 1:
            raise ValueError("propagation_steps must be at least 1")
        if convergence_tol < 0.0:
            raise ValueError("convergence_tol must be non-negative")
        if decision_normalization not in {"none", "program"}:
            raise ValueError("decision_normalization must be 'none' or 'program'")

        self.num_iters = num_iters
        self.tnorm = tnorm
        self.tnorm_wfs = tnorm_wfs if tnorm_wfs is not None else tnorm
        self.hidden_dim = hidden_dim
        self.logical_dimension = logical_dimension
        self.eps = eps
        self.propagation_steps = propagation_steps
        self.use_convergence = use_convergence
        self.convergence_tol = convergence_tol
        self.weak_negation_falsity = weak_negation_falsity
        self.decision_normalization = decision_normalization

        self.decision_rnn = nn.GRU(input_size=3, hidden_size=hidden_dim, batch_first=False)
        self.decision_head = nn.Linear(hidden_dim, 1)

    def _combine_tnorm(self, a: Tensor, b: Tensor, tnorm: str) -> Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        if b.dim() == 1:
            b = b.unsqueeze(-1)
        if tnorm == "product":
            return a * b
        if tnorm == "lukasiewicz":
            return torch.clamp(a + b - 1.0, 0.0, 1.0)
        if tnorm == "godel":
            return torch.minimum(a, b)
        raise ValueError(f"Unsupported tnorm '{tnorm}'")

    def _combine_tconorm(self, a: Tensor, b: Tensor, tnorm: str) -> Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        if b.dim() == 1:
            b = b.unsqueeze(-1)
        if tnorm == "product":
            return 1.0 - (1.0 - a) * (1.0 - b)
        if tnorm == "lukasiewicz":
            return torch.clamp(a + b, 0.0, 1.0)
        if tnorm == "godel":
            return torch.maximum(a, b)
        raise ValueError(f"Unsupported tconorm '{tnorm}'")

    def _aggregate_literals(self, values: Tensor, rule_index: Tensor, dim_size: int) -> Tensor:
        if values.numel() == 0:
            return values.new_ones(dim_size, self.logical_dimension)
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        if self.tnorm == "product":
            clamped = values.clamp(self.eps, 1.0 - self.eps)
            log_vals = torch.log(clamped)
            acc = scatter_add(log_vals, rule_index, dim=0, dim_size=dim_size)
            return torch.exp(acc)
        if self.tnorm == "lukasiewicz":
            sums = scatter_add(values, rule_index, dim=0, dim_size=dim_size)
            counts = scatter_add(torch.ones_like(values), rule_index, dim=0, dim_size=dim_size)
            return sums - (counts - 1.0)
        if self.tnorm == "godel":
            minima = torch.ones(
                dim_size,
                self.logical_dimension,
                device=values.device,
                dtype=values.dtype,
            )
            if values.numel() > 0:
                expanded_idx = rule_index.unsqueeze(-1).expand(-1, self.logical_dimension)
                minima = minima.scatter_reduce(
                    0,
                    expanded_idx,
                    values,
                    reduce="amin",
                    include_self=True,
                )
            return minima
        raise ValueError(f"Unsupported tnorm '{self.tnorm}'")

    def _aggregate_conorm(self, values: Tensor, index: Tensor, dim_size: int) -> Tensor:
        if dim_size == 0:
            return values.new_empty(dim_size, self.logical_dimension)
        if values.numel() == 0:
            return torch.zeros(
                dim_size,
                self.logical_dimension,
                device=values.device,
                dtype=values.dtype,
            )
        if values.dim() == 1:
            values = values.unsqueeze(-1)

        if self.tnorm == "product":
            one_minus = (1.0 - values).clamp(self.eps, 1.0)
            log_terms = torch.log(one_minus)
            log_acc = scatter_add(log_terms, index, dim=0, dim_size=dim_size)
            support = 1.0 - torch.exp(log_acc)
            return support

        if self.tnorm == "lukasiewicz":
            summed = scatter_add(values, index, dim=0, dim_size=dim_size)
            return torch.clamp(summed, 0.0, 1.0)

        if self.tnorm == "godel":
            base = torch.full(
                (dim_size, self.logical_dimension),
                -float("inf"),
                device=values.device,
                dtype=values.dtype,
            )
            expanded_idx = index.unsqueeze(-1).expand(-1, self.logical_dimension)
            base = base.scatter_reduce(0, expanded_idx, values, reduce="amax", include_self=True)
            base = torch.where(base == -float("inf"), torch.zeros_like(base), base)
            return base

        raise ValueError(f"Unsupported tconorm '{self.tnorm}'")

    def _aggregate_heads(
        self,
        rule_values: Tensor,
        head_edge: Tensor,
        num_atoms: int,
        tnorm: str,
    ) -> Tensor:
        if num_atoms == 0 or rule_values.numel() == 0 or head_edge.numel() == 0:
            return torch.zeros(
                num_atoms,
                self.logical_dimension,
                device=rule_values.device,
                dtype=rule_values.dtype,
            )

        if rule_values.dim() == 1:
            rule_values = rule_values.unsqueeze(-1)
        rule_idx = head_edge[0]
        atom_idx = head_edge[1]
        messages = rule_values[rule_idx]
        ones = torch.ones_like(messages)
        counts = scatter_add(ones, atom_idx, dim=0, dim_size=num_atoms)

        if tnorm == "godel":
            base = torch.full(
                (num_atoms, self.logical_dimension),
                -float("inf"),
                device=messages.device,
                dtype=messages.dtype,
            )
            expanded_idx = atom_idx.unsqueeze(-1).expand(-1, self.logical_dimension)
            base = base.scatter_reduce(0, expanded_idx, messages, reduce="amax", include_self=True)
            base = torch.where(base == -float("inf"), torch.zeros_like(base), base)
            return base

        if tnorm == "product":
            one_minus = (1.0 - messages).clamp(self.eps, 1.0)
            log_terms = torch.log(one_minus)
            log_acc = scatter_add(log_terms, atom_idx, dim=0, dim_size=num_atoms)
            support = 1.0 - torch.exp(log_acc)
            support = torch.where(counts == 0, torch.zeros_like(support), support)
            return support

        if tnorm == "lukasiewicz":
            summed = scatter_add(messages, atom_idx, dim=0, dim_size=num_atoms)
            summed = torch.where(counts == 0, torch.zeros_like(summed), summed)
            return torch.clamp(summed, 0.0, 1.0)

        raise ValueError(f"Unsupported tconorm '{tnorm}'")

    def _aggregate_heads_tnorm(
        self,
        rule_values: Tensor,
        head_edge: Tensor,
        num_atoms: int,
        tnorm: str,
    ) -> Tensor:
        if num_atoms == 0:
            return rule_values.new_empty(num_atoms, self.logical_dimension)

        if rule_values.numel() == 0 or head_edge.numel() == 0:
            return torch.ones(
                num_atoms,
                self.logical_dimension,
                device=rule_values.device,
                dtype=rule_values.dtype,
            )

        if rule_values.dim() == 1:
            rule_values = rule_values.unsqueeze(-1)
        rule_idx = head_edge[0]
        atom_idx = head_edge[1]
        messages = rule_values[rule_idx]
        ones = torch.ones_like(messages)
        counts = scatter_add(ones, atom_idx, dim=0, dim_size=num_atoms)

        if tnorm == "product":
            clamped = messages.clamp(self.eps, 1.0 - self.eps)
            log_vals = torch.log(clamped)
            acc = scatter_add(log_vals, atom_idx, dim=0, dim_size=num_atoms)
            return torch.exp(acc)

        if tnorm == "lukasiewicz":
            sums = scatter_add(messages, atom_idx, dim=0, dim_size=num_atoms)
            return torch.clamp(sums - (counts - 1.0), 0.0, 1.0)

        if tnorm == "godel":
            minima = torch.ones(
                num_atoms,
                self.logical_dimension,
                device=messages.device,
                dtype=messages.dtype,
            )
            expanded_idx = atom_idx.unsqueeze(-1).expand(-1, self.logical_dimension)
            minima = minima.scatter_reduce(
                0,
                expanded_idx,
                messages,
                reduce="amin",
                include_self=True,
            )
            return minima

        raise ValueError(f"Unsupported tnorm '{tnorm}'")

    def _apply_fact_overrides(
        self,
        rule_values: Tensor,
        fact_mask: Optional[Tensor],
        fact_values: Optional[Tensor],
        propagate_fuzzy_facts: bool = True,
    ) -> Tensor:
        if fact_mask is None:
            return rule_values

        if fact_mask.numel() != rule_values.size(0):
            raise ValueError(
                f"fact_mask size ({fact_mask.numel()}) must match num_rules ({rule_values.size(0)})"
            )
        mask = fact_mask.to(dtype=torch.bool, device=rule_values.device).view(-1)
        if not mask.any():
            return rule_values

        values = rule_values.clone()
        if fact_values is None:
            values[mask] = 1.0
            return values

        fv = fact_values.to(device=rule_values.device, dtype=rule_values.dtype)
        if fv.dim() == 1:
            fv = fv.unsqueeze(-1)
        if fv.size(-1) == 1 and self.logical_dimension > 1:
            fv = fv.expand(-1, self.logical_dimension)
        if fv.size(0) == values.size(0):
            if propagate_fuzzy_facts:
                values[mask] = fv[mask]
            else:
                values[mask] = 1.0
        elif fv.size(0) == int(mask.sum().item()):
            if propagate_fuzzy_facts:
                values[mask] = fv
            else:
                values[mask] = 1.0
        else:
            raise ValueError("fact_values must match num_rules or number of facts set in fact_mask")
        return values

    def _rule_values(
        self,
        tau: Tensor,
        phi: Tensor,
        edge_pos: Tensor,
        edge_neg: Tensor,
        num_rules: int,
        fact_mask: Optional[Tensor],
        fact_values: Optional[Tensor],
        propagate_fuzzy_facts: bool = True,
    ) -> Tensor:
        if num_rules == 0:
            return tau.new_empty(0, self.logical_dimension)

        body = torch.ones(num_rules, self.logical_dimension, device=tau.device, dtype=tau.dtype)
        if edge_pos.numel() > 0:
            pos_vals = tau[edge_pos[0]]
            body = self._aggregate_literals(pos_vals, edge_pos[1], num_rules)

        if edge_neg.numel() > 0:
            neg_vals = phi[edge_neg[0]]
            neg_support = self._aggregate_literals(neg_vals, edge_neg[1], num_rules)
            body = self._combine_tnorm(body, neg_support, self.tnorm)
        body = self._apply_fact_overrides(body, fact_mask, fact_values, propagate_fuzzy_facts)
        return _clamp01(body)


    def _rule_body_falsity(
        self,
        tau: Tensor,
        phi: Tensor,
        edge_pos: Tensor,
        edge_neg: Tensor,
        num_rules: int,
        fact_mask: Optional[Tensor],
        fact_values: Optional[Tensor],
    ) -> Tensor:
        if num_rules == 0:
            return tau.new_empty(0, self.logical_dimension)

        pos_false = torch.zeros(
            num_rules,
            self.logical_dimension,
            device=tau.device,
            dtype=tau.dtype,
        )
        if edge_pos.numel() > 0:
            pos_vals = phi[edge_pos[0]]
            pos_false = self._aggregate_conorm(pos_vals, edge_pos[1], num_rules)

        neg_false = torch.zeros(
            num_rules,
            self.logical_dimension,
            device=tau.device,
            dtype=tau.dtype,
        )
        if edge_neg.numel() > 0:
            neg_vals = tau[edge_neg[0]]
            neg_false = self._aggregate_conorm(neg_vals, edge_neg[1], num_rules)

        falsity = self._combine_tconorm(pos_false, neg_false, self.tnorm)
        if fact_mask is not None:
            if fact_mask.numel() != num_rules:
                raise ValueError(
                    f"fact_mask size ({fact_mask.numel()}) must match num_rules ({num_rules})"
                )
            mask = fact_mask.to(dtype=torch.bool, device=falsity.device).view(-1)
            if mask.any():
                falsity = falsity.clone()
                if fact_values is None:
                    falsity[mask] = 0.0
                else:
                    fv = fact_values.to(device=falsity.device, dtype=falsity.dtype)
                    if fv.dim() == 1:
                        fv = fv.unsqueeze(-1)
                    if fv.size(-1) == 1 and self.logical_dimension > 1:
                        fv = fv.expand(-1, self.logical_dimension)
                    if fv.size(0) == falsity.size(0):
                        falsity[mask] = 1.0 - fv[mask]
                    elif fv.size(0) == int(mask.sum().item()):
                        falsity_masked = fv
                        if falsity_masked.dim() == 1:
                            falsity_masked = falsity_masked.unsqueeze(-1)
                        if falsity_masked.size(-1) == 1 and self.logical_dimension > 1:
                            falsity_masked = falsity_masked.expand(-1, self.logical_dimension)
                        falsity[mask] = 1.0 - falsity_masked
                    else:
                        raise ValueError(
                            "fact_values must match num_rules or number of facts set in fact_mask"
                        )
        return _clamp01(falsity)

    def _normalize_decision_logits(self, logits: Tensor, atom_ptr: Optional[Tensor]) -> Tensor:
        if self.decision_normalization == "none" or logits.numel() <= 1:
            return logits

        normalized = logits.clone()
        if atom_ptr is None or atom_ptr.numel() <= 1:
            atom_segments = [(0, logits.size(0))]
        else:
            ptr = atom_ptr.to(device=logits.device, dtype=torch.long)
            atom_segments = [(int(ptr[i].item()), int(ptr[i + 1].item())) for i in range(ptr.numel() - 1)]

        for start, end in atom_segments:
            if end - start <= 1:
                continue
            segment = logits[start:end]
            mean = segment.mean(dim=0, keepdim=True)
            var = segment.var(dim=0, keepdim=True, unbiased=False)
            normalized[start:end] = (segment - mean) * torch.rsqrt(var + 1e-5)
        return normalized

    def _decision_logits(
        self,
        tau: Tensor,
        phi: Tensor,
        m: Tensor,
        hidden: Tensor,
        atom_ptr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        stacked = torch.stack([tau, phi, m], dim=-1)
        flat = stacked.view(-1, 3)
        rnn_in = flat.unsqueeze(0)
        rnn_out, next_hidden = self.decision_rnn(rnn_in, hidden)
        rnn_flat = rnn_out.squeeze(0)
        logits = self.decision_head(rnn_flat)
        logits = logits.view(tau.size(0), self.logical_dimension)
        logits = self._normalize_decision_logits(logits, atom_ptr)
        return logits, next_hidden

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
        fact_mask: Optional[Tensor] = None,
        fact_values: Optional[Tensor] = None,
        return_iterates: bool = False,
        return_debug: bool = False,
        max_iters: Optional[int] = None,
        initial_hidden: Optional[Tensor] = None,
    ) -> Tensor | Tuple[Tensor, Tensor] | Tuple[Tensor, dict[str, Tensor]]:
        if num_atoms != p0.size(0):
            raise ValueError(f"num_atoms ({num_atoms}) does not match p0 size ({p0.size(0)})")
        if p0.dim() == 1:
            tau = p0.unsqueeze(-1).expand(num_atoms, self.logical_dimension).clone()
            phi = torch.zeros_like(tau)
        elif p0.dim() == 2:
            if p0.size(1) == 1:
                tau = p0.expand(num_atoms, self.logical_dimension).clone()
                phi = torch.zeros_like(tau)
            elif p0.size(1) >= 2:
                tau = p0[:, 0].unsqueeze(-1).expand(num_atoms, self.logical_dimension).clone()
                phi = p0[:, 1].unsqueeze(-1).expand(num_atoms, self.logical_dimension).clone()
            else:
                raise ValueError("p0 must contain at least one column for tau.")
        else:
            raise ValueError("p0 must be a 1D or 2D tensor")

        tau = _clamp01(tau)
        phi = _clamp01(phi)

        if fact_mask is not None and fact_mask.numel() > 0:
            fact_rule_values = tau.new_zeros(num_rules, self.logical_dimension)
            fact_rule_values = self._apply_fact_overrides(
                fact_rule_values,
                fact_mask,
                fact_values,
                propagate_fuzzy_facts=True,
            )
            fact_support = self._aggregate_heads(
                fact_rule_values,
                head_edge,
                num_atoms,
                self.tnorm_wfs,
            )
            fact_support = _clamp01(fact_support)
            tau = torch.maximum(tau, fact_support)
            has_fact = fact_support > 0.0
            phi = torch.where(has_fact, 1.0 - tau, torch.zeros_like(phi))

        collect_iterates = return_iterates or return_debug
        iterates: list[Tensor] = []
        tau_traces: list[Tensor] = []
        phi_traces: list[Tensor] = []
        if collect_iterates:
            probs0 = 0.5 * (tau + (1.0 - phi))
            iterates.append(probs0.clone())
            tau_traces.append(tau.clone())
            phi_traces.append(phi.clone())

        debug: dict[str, Tensor] = {}
        limit = max_iters if max_iters is not None else self.num_iters
        if limit < 1:
            raise ValueError("max_iters must be at least 1")

        if initial_hidden is not None:
            if initial_hidden.dim() == 2:
                if initial_hidden.size(0) == self.hidden_dim:
                    initial_hidden = initial_hidden.transpose(0, 1)
                initial_hidden = initial_hidden.unsqueeze(0)
            elif initial_hidden.dim() == 3 and initial_hidden.size(0) == 1:
                if (
                    initial_hidden.size(1) == self.hidden_dim
                    and initial_hidden.size(2) != self.hidden_dim
                ):
                    initial_hidden = initial_hidden.transpose(1, 2)
            if (
                initial_hidden.dim() != 3
                or initial_hidden.size(0) != 1
                or initial_hidden.size(2) != self.hidden_dim
            ):
                raise ValueError("initial_hidden must be shaped [1, num_slots, hidden_dim].")
            controller_state = initial_hidden.to(device=tau.device, dtype=tau.dtype)
            if controller_state.size(1) == num_atoms:
                controller_state = controller_state.repeat(1, self.logical_dimension, 1)
            if controller_state.size(1) != num_atoms * self.logical_dimension:
                raise ValueError(
                    "initial_hidden slot count must equal num_atoms "
                    "or num_atoms * logical_dimension."
                )
        else:
            controller_state = torch.rand(
                1,
                num_atoms * self.logical_dimension,
                self.hidden_dim,
                device=tau.device,
            )

        prev_tau: Optional[Tensor] = None
        prev_phi: Optional[Tensor] = None

        for _ in range(self.propagation_steps):
            prev_tau_inner = tau.clone()
            rule_vals = self._rule_values(
                tau,
                phi,
                edge_pos,
                edge_neg,
                num_rules,
                fact_mask,
                fact_values,
            )
            tau = self._aggregate_heads(rule_vals, head_edge, num_atoms, self.tnorm)
            delta_inner_tau = torch.max(tau - prev_tau_inner).item()

            if not self.weak_negation_falsity:
                prev_phi_inner = phi.clone()
                rule_falsity = self._rule_body_falsity(
                    tau,
                    phi,
                    edge_pos,
                    edge_neg,
                    num_rules,
                    fact_mask,
                    fact_values,
                )
                phi = self._aggregate_heads_tnorm(
                    rule_falsity,
                    head_edge,
                    num_atoms,
                    self.tnorm_wfs,
                )
                delta_inner_phi = torch.max(torch.abs(phi - prev_phi_inner)).item()
                delta_inner_tau = max(delta_inner_tau, delta_inner_phi)

            if delta_inner_tau <= self.convergence_tol:
                tau = prev_tau_inner
                break

        for _ in range(limit):
            if self.use_convergence:
                prev_tau = tau.clone()
                prev_phi = phi.clone()

            m = self._combine_tnorm(1 - tau, 1 - phi, self.tnorm)
            logits, controller_state = self._decision_logits(
                tau,
                phi,
                m,
                controller_state,
                atom_ptr=atom_ptr,
            )
            masked_logits = logits
            pi = torch.sigmoid(masked_logits)
            phi = _clamp01(phi + m * pi)

            for _ in range(self.propagation_steps):
                prev_tau_inner = tau.clone()
                rule_vals = self._rule_values(
                    tau,
                    phi,
                    edge_pos,
                    edge_neg,
                    num_rules,
                    fact_mask,
                    fact_values,
                )
                tau = self._aggregate_heads(rule_vals, head_edge, num_atoms, self.tnorm)
                delta_inner_tau = torch.max(tau - prev_tau_inner).item()

                if delta_inner_tau <= self.convergence_tol:
                    tau = prev_tau_inner
                    break

            probs = 0.5 * (tau + (1.0 - phi))

            if collect_iterates:
                iterates.append(probs.clone())
                tau_traces.append(tau.clone())
                phi_traces.append(phi.clone())

            if self.use_convergence and prev_tau is not None and prev_phi is not None:
                delta_tau = torch.max(torch.abs(tau - prev_tau)).item()
                delta_phi = torch.max(torch.abs(phi - prev_phi)).item()
                if max(delta_tau, delta_phi) <= self.convergence_tol:
                    break

        selected_dims_tensor: Optional[Tensor] = None
        if self.logical_dimension == 1:
            tau_projected = tau.view(-1)
            phi_projected = phi.view(-1)
            probs_out = probs.view(-1)
        else:
            diff_tensor = torch.abs(0.5 - probs)
            if atom_ptr is not None and atom_ptr.numel() > 1:
                ptr = atom_ptr.to(device=tau.device, dtype=torch.long)
                atom_segments = [
                    (int(ptr[i].item()), int(ptr[i + 1].item()))
                    for i in range(ptr.numel() - 1)
                ]
            else:
                atom_segments = [(0, num_atoms)]

            num_segments = max(1, len(atom_segments))
            segment_lengths = torch.tensor(
                [max(0, end - start) for start, end in atom_segments],
                dtype=torch.long,
                device=tau.device,
            )
            if int(segment_lengths.sum().item()) != num_atoms:
                segment_lengths = torch.tensor([num_atoms], device=tau.device, dtype=torch.long)

            segment_ids = torch.repeat_interleave(
                torch.arange(num_segments, device=tau.device),
                segment_lengths,
            )
            if segment_ids.numel() < num_atoms:
                pad = torch.full(
                    (num_atoms - segment_ids.numel(),),
                    num_segments - 1,
                    device=tau.device,
                    dtype=torch.long,
                )
                segment_ids = torch.cat([segment_ids, pad], dim=0)
            elif segment_ids.numel() > num_atoms:
                segment_ids = segment_ids[:num_atoms]

            expanded_ids = segment_ids.unsqueeze(-1).expand(-1, self.logical_dimension)
            base = torch.full(
                (num_segments, self.logical_dimension),
                float("inf"),
                device=tau.device,
                dtype=tau.dtype,
            )
            min_vals = base.scatter_reduce(
                0,
                expanded_ids,
                diff_tensor,
                reduce="amin",
                include_self=True,
            )
            selected_dims_tensor = torch.argmax(min_vals, dim=1)

            if segment_lengths.numel() > 0 and int(segment_lengths.sum().item()) > 0:
                atom_dim_indices = torch.repeat_interleave(
                    selected_dims_tensor[: segment_lengths.numel()],
                    segment_lengths,
                    dim=0,
                )
            else:
                atom_dim_indices = torch.empty(0, dtype=torch.long, device=tau.device)
            if atom_dim_indices.numel() < num_atoms:
                pad_dim = selected_dims_tensor[-1:].expand(num_atoms - atom_dim_indices.numel())
                atom_dim_indices = torch.cat([atom_dim_indices, pad_dim], dim=0)
            elif atom_dim_indices.numel() > num_atoms:
                atom_dim_indices = atom_dim_indices[:num_atoms]

            def _project_tensor(tensor: Tensor) -> Tensor:
                if (
                    tensor.dim() <= 1
                    or tensor.size(0) != num_atoms
                    or tensor.size(1) != self.logical_dimension
                ):
                    if tensor.dim() > 1 and tensor.size(-1) == 1:
                        return tensor.squeeze(-1)
                    return tensor
                gathered = torch.gather(tensor, 1, atom_dim_indices.unsqueeze(-1))
                return gathered.squeeze(-1)

            tau_projected = _project_tensor(tau)
            phi_projected = _project_tensor(phi)
            probs_out = _project_tensor(probs)

        if return_debug:
            debug["tau_projected"] = tau_projected.detach()
            debug["phi_projected"] = phi_projected.detach()
            debug["decision_logits"] = logits.detach()
            debug["decision_pi"] = pi.detach()
            debug["probabilities"] = (
                torch.stack(iterates, dim=0).detach() if iterates else torch.empty(0)
            )
            debug["tau_iterates"] = (
                torch.stack(tau_traces, dim=0).detach() if tau_traces else torch.empty(0)
            )
            debug["phi_iterates"] = (
                torch.stack(phi_traces, dim=0).detach() if phi_traces else torch.empty(0)
            )
            if selected_dims_tensor is not None:
                debug["selected_dimensions"] = selected_dims_tensor.detach()
            return probs_out, debug

        if return_iterates:
            iter_flat = (
                torch.stack(iterates, dim=0)
                if iterates
                else torch.empty(0, device=probs_out.device)
            )
            return probs_out, iter_flat
        return probs_out
