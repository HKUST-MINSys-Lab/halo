"""Rung 4 — labelled adaptation with fine-tuning allowed, every treatment applied to every model.

Ladder by cost: linear probe → small classifier (both on cached features, all six providers) →
LoRA → full fine-tune (raw windows; HALO and the released trunks under the encoder contract) →
a from-scratch specialist on the same k windows. Scored on rung 2's execution split so the
enrollment-vs-fine-tuning crossover is measured on identical windows. Design: roadmap, rung 4.
"""
