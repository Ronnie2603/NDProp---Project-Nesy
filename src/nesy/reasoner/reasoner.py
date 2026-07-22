from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ndprop import NDProp

from nesy.programs.base import BatchProgram, ProgramSpec, QueryProgramSpec


class NeSyReasoner(nn.Module):
    """
    Generic wrapper that injects observed evidence into a grounded logic program
    and evaluates it with the differentiable stable model approximator.
    """

    def __init__(self, program: ProgramSpec, model: NDProp, *, init_hidden: bool = False) -> None:
        super().__init__()
        self.program = program
        self.model = model
        self.last_tau_phi: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self.use_hidden_init = init_hidden
        self.hidden_gen: Optional[nn.Linear] = None
        if init_hidden:
            total_inputs = sum(v.numel() for v in program.input_fact_rules.values())
            hidden_dim = getattr(model, "hidden_dim", None)
            logical_dim = getattr(model, "logical_dimension", 1)
            if hidden_dim is not None and hidden_dim > 0 and total_inputs > 0:
                self.hidden_gen = nn.Sequential(
                    nn.Linear(total_inputs, hidden_dim * program.num_atoms * logical_dim),
                    nn.Tanh()
                )

    def forward(
        self,
        evidence: Dict[str, torch.Tensor],
        *,
        initial_hidden: Optional[torch.Tensor] = None,
        queries: Optional[torch.Tensor] = None,
        return_iterates: bool = False,
        return_tau_phi: bool = False,
    ):
        if not evidence:
            raise ValueError("Evidence dictionary must not be empty.")
        first = next(iter(evidence.values()))
        if first.dim() == 1:
            raise ValueError("Evidence tensors must be batched.")
        batch_size = first.size(0)
        device = first.device
        dtype = first.dtype

        logical_dim = getattr(self.model, "logical_dimension", 1)
        if isinstance(self.program, QueryProgramSpec):
            if queries is None:
                raise ValueError("queries must be provided for QueryProgramSpec.")
            if queries.dim() != 1 or queries.size(0) != batch_size:
                raise ValueError("queries must be a 1D tensor matching batch size.")
            base_batch = self.program.get_batch_for_queries(queries, device)
        else:
            base_batch = self.program.get_batch(batch_size, device)
        if not self.model.weak_negation_falsity:
            batch_graph = self._inject_evidence(base_batch, evidence, dtype)
        else:
            batch_graph = base_batch

        seed = torch.zeros(batch_graph.num_atoms, logical_dim, device=device, dtype=dtype)
        if initial_hidden is None and self.hidden_gen is not None:
            flat_evidence: list[torch.Tensor] = []
            for key, rules in self.program.input_fact_rules.items():
                if rules.numel() == 0:
                    continue
                if key not in evidence:
                    raise ValueError(f"Missing evidence for '{key}' needed for hidden init.")
                vals = evidence[key].to(device=device, dtype=dtype).reshape(batch_size, -1)
                flat_evidence.append(vals)
            if flat_evidence:
                concat = torch.cat(flat_evidence, dim=1)
                hidden_flat = self.hidden_gen(concat)
                hidden_flat = hidden_flat.view(batch_size, self.program.num_atoms * logical_dim, -1)
                initial_hidden = hidden_flat.view(1, batch_size * self.program.num_atoms * logical_dim, -1)
        need_debug = return_tau_phi
        outputs = self.model(
            seed,
            batch_graph.edge_pos,
            batch_graph.edge_neg,
            batch_graph.head_edge,
            batch_graph.num_atoms,
            batch_graph.num_rules,
            atom_ptr=batch_graph.atom_ptr,
            fact_mask=batch_graph.fact_mask,
            fact_values=batch_graph.fact_values,
            return_iterates=return_iterates,
            return_debug=need_debug,
            initial_hidden=initial_hidden,
        )

        iterates = None
        debug = None
        if need_debug:
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise RuntimeError("Expected (preds, debug) when return_debug=True.")
            preds, debug = outputs
        elif return_iterates:
            preds, iterates = outputs
        else:
            preds = outputs

        total_atoms = batch_size * self.program.num_atoms
        if preds.numel() % total_atoms != 0:
            raise RuntimeError(
                "Predictions size mismatch: "
                f"got {preds.numel()} values for batch={batch_size}, atoms={self.program.num_atoms}."
            )
        logical_dim_out = preds.numel() // total_atoms
        assignments = preds.view(batch_size, self.program.num_atoms, logical_dim_out)
        if logical_dim_out == 1:
            assignments = assignments.squeeze(-1)

        if iterates is not None:
            iterates = iterates.view(iterates.size(0), batch_size, self.program.num_atoms, logical_dim_out)
            if logical_dim_out == 1:
                iterates = iterates.squeeze(-1)

        if debug is not None:
            tau_proj = debug.get("tau_projected")
            phi_proj = debug.get("phi_projected")
            if tau_proj is not None and phi_proj is not None:
                self.last_tau_phi = (
                    tau_proj.to(device=assignments.device).view(batch_size, self.program.num_atoms, logical_dim_out),
                    phi_proj.to(device=assignments.device).view(batch_size, self.program.num_atoms, logical_dim_out),
                )

        if return_iterates:
            return assignments, iterates
        return assignments

    def _inject_evidence(
        self,
        batch_graph: BatchProgram,
        evidence: Dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> BatchProgram:
        fact_values = batch_graph.fact_values.to(device=next(iter(evidence.values())).device, dtype=dtype)
        for key, rules in batch_graph.input_fact_rules.items():
            if rules.numel() == 0:
                continue
            if key not in evidence:
                raise ValueError(f"Missing evidence for '{key}'.")
            values = evidence[key].to(device=fact_values.device, dtype=fact_values.dtype)
            flat = values.reshape(-1)
            if flat.numel() != rules.numel():
                raise ValueError(f"Evidence for '{key}' has {flat.numel()} values but expected {rules.numel()}.")
            fact_values = fact_values.scatter(0, rules.to(device=fact_values.device), flat)
        return batch_graph.with_fact_values(fact_values)
