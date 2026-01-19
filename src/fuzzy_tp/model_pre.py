from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add


TNorm = Literal["product", "lukasiewicz", "godel"]
TConorm = Literal["prob_sum", "lukasiewicz", "godel"]


def _clamp01(x: Tensor) -> Tensor:
    return torch.clamp(x, 0.0, 1.0)


class DecisionBasedGNN(nn.Module):
    """
    Continuous relaxation of the decision-based partial update semantics.

    Each atom keeps (tau, phi) in [0,1]. Propagation is deterministic: K add-only
    steps that relax the completion fixpoint with tau <- tau oplus u and
    phi <- phi oplus w (w aggregates failure of all rule bodies). A separate
    decision controller (RNN_dec) maintains state s^t across outer iterations,
    produces logits over atoms, and a masked softmax biases toward undecided
    atoms m = (1 - tau) otimes (1 - phi). Phi is increased by m odot pi; final belief is
    p = 0.5 * (tau + (1 - phi)). In the binary, exact-propagation limit this
    matches the discrete transition system one-to-one.
    """

    def __init__(
        self,
        *,
        num_iters: int = 30,
        tnorm: TNorm = "product",
        tconorm: TConorm = "prob_sum",
        hidden_dim: int = 64,
        logical_dimension: int = 1,
        eps: float = 1e-6,
        propagation_steps: int = 10,
        use_convergence: bool = True,
        convergence_tol: float = 1e-4,
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
        if convergence_tol <= 0.0:
            raise ValueError("convergence_tol must be positive")

        self.num_iters = num_iters
        self.tnorm = tnorm
        self.tconorm = tconorm
        self.hidden_dim = hidden_dim
        self.logical_dimension = logical_dimension
        self.eps = eps
        self.propagation_steps = propagation_steps
        self.use_convergence = use_convergence
        self.convergence_tol = convergence_tol

        # Decision RNN operates on flattened (atom, logical_dimension) pairs and
        # carries the controller state across outer iterations.
        self.decision_rnn = nn.GRU(input_size=3, hidden_size=hidden_dim, batch_first=False)
        self.decision_head = nn.Linear(hidden_dim, 1)
        # Per-dimension weight head (sigmoid-applied each step) to relax the simplex constraint on pi.
        self.decision_weight_head = nn.Linear(hidden_dim, 1)

    # ------------------------------------------------------------------
    # Literal evaluation and aggregation
    # ------------------------------------------------------------------
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
            minima = torch.ones(dim_size, self.logical_dimension, device=values.device, dtype=values.dtype)
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

    def _apply_fact_overrides(
        self,
        rule_values: Tensor,
        fact_mask: Optional[Tensor],
        fact_values: Optional[Tensor],
    ) -> Tensor:
        if fact_mask is None:
            if fact_values is not None:
                raise ValueError("fact_values provided but fact_mask is None")
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
            values[fv > 0] = fv[fv > 0]
        elif fv.size(0) == int(mask.sum().item()):
            values[mask] = fv
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
    ) -> Tensor:
        if num_rules == 0:
            return tau.new_empty(0, self.logical_dimension)

        body = torch.ones(num_rules, self.logical_dimension, device=tau.device, dtype=tau.dtype)
        if edge_pos.numel() > 0:
            if edge_pos[0].max().item() >= tau.size(0):
                raise IndexError(
                    f"edge_pos atom index {edge_pos[0].max().item()} exceeds tau size {tau.size(0)}"
                )
            pos_vals = tau[edge_pos[0]]
            body = self._aggregate_literals(pos_vals, edge_pos[1], num_rules)

        if edge_neg.numel() > 0:
            if edge_neg[0].max().item() >= phi.size(0):
                raise IndexError(
                    f"edge_neg atom index {edge_neg[0].max().item()} exceeds phi size {phi.size(0)}"
                )
            neg_vals = phi[edge_neg[0]]
            neg_support = self._aggregate_literals(neg_vals, edge_neg[1], num_rules)
            body = self._combine_tnorm(body, neg_support)
        body = self._apply_fact_overrides(body, fact_mask, fact_values)
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

        pos_false = torch.zeros(num_rules, self.logical_dimension, device=tau.device, dtype=tau.dtype)
        if edge_pos.numel() > 0:
            pos_vals = phi[edge_pos[0]]
            pos_false = self._aggregate_conorm(pos_vals, edge_pos[1], num_rules)

        neg_false = torch.zeros(num_rules, self.logical_dimension, device=tau.device, dtype=tau.dtype)
        if edge_neg.numel() > 0:
            neg_vals = tau[edge_neg[0]]
            neg_false = self._aggregate_conorm(neg_vals, edge_neg[1], num_rules)

        falsity = self._combine_tconorm(pos_false, neg_false)
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
                        raise ValueError("fact_values must match num_rules or number of facts set in fact_mask")
        return _clamp01(falsity)

    def _combine_tnorm(self, a: Tensor, b: Tensor) -> Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        if b.dim() == 1:
            b = b.unsqueeze(-1)
        if self.tnorm == "product":
            return a * b
        if self.tnorm == "lukasiewicz":
            return torch.clamp(a + b - 1.0, 0.0, 1.0)
        if self.tnorm == "godel":
            return torch.minimum(a, b)
        raise ValueError(f"Unsupported tnorm '{self.tnorm}'")

    def _combine_tconorm(self, a: Tensor, b: Tensor) -> Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        if b.dim() == 1:
            b = b.unsqueeze(-1)
        if self.tconorm == "prob_sum":
            return 1.0 - (1.0 - a) * (1.0 - b)
        if self.tconorm == "lukasiewicz":
            return torch.clamp(a + b, 0.0, 1.0)
        if self.tconorm == "godel":
            return torch.maximum(a, b)
        raise ValueError(f"Unsupported tconorm '{self.tconorm}'")

    def _aggregate_heads(self, rule_values: Tensor, head_edge: Tensor, num_atoms: int) -> Tensor:
        if num_atoms == 0 or rule_values.numel() == 0 or head_edge.numel() == 0:
            return torch.zeros(num_atoms, self.logical_dimension, device=rule_values.device, dtype=rule_values.dtype)

        if rule_values.dim() == 1:
            rule_values = rule_values.unsqueeze(-1)
        rule_idx = head_edge[0]
        atom_idx = head_edge[1]
        messages = rule_values[rule_idx]
        ones = torch.ones_like(messages)
        counts = scatter_add(ones, atom_idx, dim=0, dim_size=num_atoms)

        if self.tconorm == "godel":
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

        if self.tconorm == "prob_sum":
            one_minus = (1.0 - messages).clamp(self.eps, 1.0)
            log_terms = torch.log(one_minus)
            log_acc = scatter_add(log_terms, atom_idx, dim=0, dim_size=num_atoms)
            support = 1.0 - torch.exp(log_acc)
            support = torch.where(counts == 0, torch.zeros_like(support), support)
            return support

        if self.tconorm == "lukasiewicz":
            summed = scatter_add(messages, atom_idx, dim=0, dim_size=num_atoms)
            summed = torch.where(counts == 0, torch.zeros_like(summed), summed)
            return torch.clamp(summed, 0.0, 1.0)

        raise ValueError(f"Unsupported tconorm '{self.tconorm}'")

    def _aggregate_conorm(self, values: Tensor, index: Tensor, dim_size: int) -> Tensor:
        if dim_size == 0:
            return values.new_empty(dim_size, self.logical_dimension)
        if values.numel() == 0:
            return torch.zeros(dim_size, self.logical_dimension, device=values.device, dtype=values.dtype)
        if values.dim() == 1:
            values = values.unsqueeze(-1)

        if self.tconorm == "prob_sum":
            one_minus = (1.0 - values).clamp(self.eps, 1.0)
            log_terms = torch.log(one_minus)
            log_acc = scatter_add(log_terms, index, dim=0, dim_size=dim_size)
            support = 1.0 - torch.exp(log_acc)
            return support

        if self.tconorm == "lukasiewicz":
            summed = scatter_add(values, index, dim=0, dim_size=dim_size)
            return torch.clamp(summed, 0.0, 1.0)

        if self.tconorm == "godel":
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

        raise ValueError(f"Unsupported tconorm '{self.tconorm}'")

    def _aggregate_heads_tnorm(self, rule_values: Tensor, head_edge: Tensor, num_atoms: int) -> Tensor:
        if num_atoms == 0:
            return rule_values.new_empty(num_atoms, self.logical_dimension)

        if rule_values.numel() == 0 or head_edge.numel() == 0:
            return torch.ones(num_atoms, self.logical_dimension, device=rule_values.device, dtype=rule_values.dtype)

        if rule_values.dim() == 1:
            rule_values = rule_values.unsqueeze(-1)
        rule_idx = head_edge[0]
        atom_idx = head_edge[1]
        messages = rule_values[rule_idx]
        ones = torch.ones_like(messages)
        counts = scatter_add(ones, atom_idx, dim=0, dim_size=num_atoms)

        if self.tnorm == "product":
            clamped = messages.clamp(self.eps, 1.0 - self.eps)
            log_vals = torch.log(clamped)
            acc = scatter_add(log_vals, atom_idx, dim=0, dim_size=num_atoms)
            return torch.exp(acc)

        if self.tnorm == "lukasiewicz":
            sums = scatter_add(messages, atom_idx, dim=0, dim_size=num_atoms)
            return torch.clamp(sums - (counts - 1.0), 0.0, 1.0)

        if self.tnorm == "godel":
            minima = torch.ones(num_atoms, self.logical_dimension, device=messages.device, dtype=messages.dtype)
            expanded_idx = atom_idx.unsqueeze(-1).expand(-1, self.logical_dimension)
            minima = minima.scatter_reduce(
                0,
                expanded_idx,
                messages,
                reduce="amin",
                include_self=True,
            )
            return minima

        raise ValueError(f"Unsupported tnorm '{self.tnorm}'")

    # ------------------------------------------------------------------
    # Update dynamics
    # ------------------------------------------------------------------
    def _decision_logits(
        self,
        tau: Tensor,
        phi: Tensor,
        m: Tensor,
        hidden: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        stacked = torch.stack([tau, phi, m], dim=-1)
        flat = stacked.view(-1, 3)
        rnn_in = flat.unsqueeze(0)
        rnn_out, next_hidden = self.decision_rnn(rnn_in, hidden)
        rnn_flat = rnn_out.squeeze(0)
        logits = self.decision_head(rnn_flat)
        weights = self.decision_weight_head(rnn_flat)
        return (
            logits.view(tau.size(0), self.logical_dimension),
            weights.view(tau.size(0), self.logical_dimension),
            next_hidden,
        )

    def _graph_softmax(self, logits: Tensor, atom_ptr: Optional[Tensor]) -> Tensor:
        if atom_ptr is None or atom_ptr.numel() <= 1:
            return torch.softmax(logits, dim=0)

        ptr = atom_ptr.to(device=logits.device, dtype=torch.long)
        if ptr[-1].item() > logits.size(0):
            raise ValueError("atom_ptr end exceeds number of atoms.")

        # Build group id per atom from CSR-style ptr
        # Example: ptr=[0,3,5] -> group=[0,0,0,1,1]
        sizes = (ptr[1:] - ptr[:-1]).clamp_min(0)
        group = torch.repeat_interleave(
            torch.arange(sizes.numel(), device=logits.device, dtype=torch.long),
            sizes
        )

        # If ptr[-1] < logits.size(0), treat remaining atoms as their own groups or ignore.
        # Your original code leaves them as zeros in `out`; to match that behavior:
        if ptr[-1].item() < logits.size(0):
            out = torch.zeros_like(logits)
            out[:ptr[-1]] = self._scatter_softmax_1d(logits[:ptr[-1]], group)
            return out

        return self._scatter_softmax_1d(logits, group)


    def _scatter_softmax_1d(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        from torch_scatter import scatter_softmax

        if x.numel() == 0 or index.numel() == 0:
            return torch.zeros_like(x)
        x = x.contiguous()
        index = index.to(device=x.device, dtype=torch.long).contiguous()
        y = scatter_softmax(x, index=index, dim=0)
        return y

    # ------------------------------------------------------------------
    # Public forward
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
        return_stability: bool = False,
        initial_hidden: Optional[Tensor] = None,
    ) -> Tensor | Tuple[Tensor, list[Tensor]] | Tuple[Tensor, list[Tensor], Tensor]:
        if num_atoms != p0.size(0):
            raise ValueError(f"num_atoms ({num_atoms}) does not match p0 size ({p0.size(0)})")
        # Initialise tau and phi from p0; if p0 supplies two channels, use both.
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

        collect_iterates = return_iterates or return_debug
        iterates: list[Tensor] = []
        tau_traces: list[Tensor] = []
        phi_traces: list[Tensor] = []
        if collect_iterates:
            probs0 = 0.5 * (tau + (1.0 - phi))
            iterates.append(probs0.clone())
            tau_traces.append(tau.clone())
            phi_traces.append(phi.clone())

        debug: Dict[str, Tensor] = {}
        limit = max_iters if max_iters is not None else self.num_iters
        if limit < 1:
            raise ValueError("max_iters must be at least 1")

        # Decision controller state: provided by caller or zeros.
        if initial_hidden is not None:
            if initial_hidden.dim() != 3 or initial_hidden.size(0) != 1 or initial_hidden.size(2) != self.hidden_dim:
                raise ValueError("initial_hidden must be shaped [1, num_slots, hidden_dim].")
            controller_state = initial_hidden.to(device=tau.device, dtype=tau.dtype)
            if controller_state.size(1) != num_atoms * self.logical_dimension:
                raise ValueError("initial_hidden slot count must equal num_atoms * logical_dimension.")
        else:
            controller_state = torch.zeros(1, num_atoms * self.logical_dimension, self.hidden_dim, device=tau.device)

        prev_tau: Optional[Tensor] = None
        prev_phi: Optional[Tensor] = None
        # init_tau = tau.clone()
        # init_phi = phi.clone()
        pi = torch.zeros_like(tau)
        for its in range(limit):
            if self.use_convergence or return_stability:
                prev_tau = tau.clone()
                prev_phi = phi.clone()
            
            # if its == 4:
            #     tau = init_tau.clone()
            #     phi = init_phi.clone()

            # Propagation phase: iterate until convergence or max inner steps.
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
                tau = self._aggregate_heads(rule_vals, head_edge, num_atoms)
                delta_inner_tau = torch.max(tau - prev_tau_inner).item()

                if its == -1:
                    prev_phi_inner = phi.clone()
                    falsity_vals = self._rule_body_falsity(
                        tau,
                        phi,
                        edge_pos,
                        edge_neg,
                        num_rules,
                        fact_mask,
                        fact_values,
                    )
                    phi = self._aggregate_heads_tnorm(falsity_vals, head_edge, num_atoms)
                    delta_inner_phi = torch.max(phi - prev_phi_inner).item()
                    delta_inner_tau = max(delta_inner_tau, delta_inner_phi)
                    # print(delta_inner_tau)
                # print(((tau-prev_tau_inner) < 0).sum())
                # print(((phi-prev_phi_inner) < 0).sum())

                if delta_inner_tau <= self.convergence_tol:
                # if (delta_inner_tau) <= self.convergence_tol:
                    break

            # Decision phase.
            # m = (1.0 - tau)*(1.0 - phi)
            m = self._combine_tnorm(1-tau, 1-phi)  # Square to sharpen
            logits, weights, controller_state = self._decision_logits(tau, phi, m, controller_state)
            # masked_logits = logits + torch.log(m + self.eps)
            masked_logits = logits * (m)
            # pi = self._graph_softmax(masked_logits, atom_ptr)
            pi = torch.sigmoid(masked_logits)
            # weight_sig = torch.sigmoid(weights)
            # if atom_ptr is None or atom_ptr.numel() <= 1:
            #     weight_global = weight_sig.mean(dim=0, keepdim=True)
            #     weight_expanded = weight_global.expand_as(pi)
            # else:
            #     ptr = atom_ptr.to(device=weights.device, dtype=torch.long)
            #     if ptr[-1].item() > weight_sig.size(0):
            #         raise ValueError("atom_ptr end exceeds number of atoms for weights.")
            #     group_means = []
            #     for i in range(ptr.numel() - 1):
            #         start = int(ptr[i].item())
            #         end = int(ptr[i + 1].item())
            #         if end <= start:
            #             group_means.append(weight_sig.new_zeros(1, weight_sig.size(-1)))
            #         else:
            #             group_means.append(weight_sig[start:end].mean(dim=0, keepdim=True))
            #     weight_global = torch.cat(group_means, dim=0)  # [num_graphs, logical_dimension]
            #     sizes = (ptr[1:] - ptr[:-1]).clamp_min(0)
            #     weight_expanded = weight_global.repeat_interleave(sizes, dim=0)
            #     # If atom_ptr shorter than num atoms, pad zeros; if longer, truncate.
            #     if weight_expanded.size(0) < pi.size(0):
            #         pad = weight_expanded.new_zeros(pi.size(0) - weight_expanded.size(0), weight_expanded.size(1))
            #         weight_expanded = torch.cat([weight_expanded, pad], dim=0)
            #     weight_expanded = weight_expanded[: pi.size(0)]
            # pi = pi * weight_expanded  # scales each logical dimension independently, so pi need not sum to 1
            phi = _clamp01(phi + m * pi)

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
        
        
        # Final propagation to convergence for stability analysis.
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
            tau = self._aggregate_heads(rule_vals, head_edge, num_atoms)
            delta_inner_tau = torch.max(tau - prev_tau_inner).item()
            
            if delta_inner_tau <= self.convergence_tol:
                break

        prev_probs = None
        if prev_tau is not None and prev_phi is not None:
            prev_probs = 0.5 * (prev_tau + (1.0 - prev_phi))
        current_probs = 0.5 * (tau + (1.0 - phi))
        stability_tensor = torch.abs(current_probs - (prev_probs if prev_probs is not None else current_probs))
        diff_tensor = torch.abs(0.5 - current_probs)

        if atom_ptr is not None and atom_ptr.numel() > 1:
            ptr = atom_ptr.to(device=tau.device, dtype=torch.long)
            atom_segments = [(int(ptr[i].item()), int(ptr[i + 1].item())) for i in range(ptr.numel() - 1)]
        else:
            atom_segments = [(0, num_atoms)]

        selected_dims_tensor = torch.zeros(max(1, len(atom_segments)), dtype=torch.long, device=tau.device)
        for seg_idx, (start, end) in enumerate(atom_segments):
            seg_len = max(0, end - start)
            if seg_len <= 0:
                selected_dims_tensor[seg_idx] = 0
                continue
            seg_diff = diff_tensor[start:end]
            max_diff = seg_diff.min(dim=0).values
            best_dim = int(torch.argmax(max_diff).item())
            selected_dims_tensor[seg_idx] = best_dim

        segment_lengths = torch.tensor(
            [max(0, end - start) for start, end in atom_segments],
            dtype=torch.long,
            device=tau.device,
        )
        if segment_lengths.numel() > 0 and int(segment_lengths.sum().item()) > 0:
            atom_dim_indices = torch.repeat_interleave(selected_dims_tensor[:segment_lengths.numel()], segment_lengths, dim=0)
        else:
            atom_dim_indices = torch.empty(0, dtype=torch.long, device=tau.device)
        if atom_dim_indices.numel() < num_atoms:
            pad_dim = selected_dims_tensor[-1:].expand(num_atoms - atom_dim_indices.numel())
            atom_dim_indices = torch.cat([atom_dim_indices, pad_dim], dim=0)
        elif atom_dim_indices.numel() > num_atoms:
            atom_dim_indices = atom_dim_indices[:num_atoms]

        def _project_tensor(tensor: Tensor) -> Tensor:
            if tensor.dim() <= 1 or tensor.size(0) != num_atoms or tensor.size(1) != self.logical_dimension:
                return tensor.squeeze(-1) if tensor.dim() > 1 and tensor.size(-1) == 1 else tensor
            gathered = torch.gather(tensor, 1, atom_dim_indices.unsqueeze(-1))
            return gathered.squeeze(-1)

        tau_projected = _project_tensor(tau)
        phi_projected = _project_tensor(phi)
        tau_final = _project_tensor(current_probs)
        stability = _project_tensor(stability_tensor) if return_stability else None
        iterates_projected: list[Tensor] = []
        if iterates:
            iterates_projected = [_project_tensor(it) for it in iterates]
        tau_iterates_projected: list[Tensor] = []
        phi_iterates_projected: list[Tensor] = []
        if tau_traces:
            tau_iterates_projected = [_project_tensor(it) for it in tau_traces]
        if phi_traces:
            phi_iterates_projected = [_project_tensor(it) for it in phi_traces]

        if return_debug:
            debug["tau"] = tau.detach().cpu()
            debug["phi"] = phi.detach().cpu()
            debug["tau_projected"] = tau_projected.detach().cpu()
            debug["phi_projected"] = phi_projected.detach().cpu()
            debug["decision_mask"] = m.detach().cpu()
            debug["decision_logits"] = logits.detach().cpu()
            debug["decision_pi"] = pi.detach().cpu()
            # debug["decision_weight"] = weight_expanded.detach().cpu()
            debug["iterates"] = torch.stack(iterates, dim=0).detach().cpu() if iterates else torch.empty(0)
            debug["probabilities"] = torch.stack(iterates_projected, dim=0).detach().cpu() if iterates_projected else torch.empty(0)
            debug["tau_iterates"] = torch.stack(tau_traces, dim=0).detach().cpu() if tau_traces else torch.empty(0)
            debug["phi_iterates"] = torch.stack(phi_traces, dim=0).detach().cpu() if phi_traces else torch.empty(0)
            # print(phi)
            debug["tau_iterates_projected"] = (
                torch.stack(tau_iterates_projected, dim=0).detach().cpu()
                if tau_iterates_projected
                else torch.empty(0)
            )
            debug["phi_iterates_projected"] = (
                torch.stack(phi_iterates_projected, dim=0).detach().cpu()
                if phi_iterates_projected
                else torch.empty(0)
            )
            debug["selected_dimensions"] = selected_dims_tensor.detach().cpu()
            if return_stability and stability is not None:
                debug["stability"] = stability.detach().cpu()
            return tau_final, debug

        if return_iterates:
            iter_flat = iterates_projected if iterates_projected else []
            if return_stability and stability is not None:
                return tau_final, iter_flat, stability
            return tau_final, iter_flat

        if return_stability and stability is not None:
            return tau_final, stability
        return tau_final


# Convenience alias used by other modules when opting into the new semantics.
DecisionBasedStableModel = DecisionBasedGNN
