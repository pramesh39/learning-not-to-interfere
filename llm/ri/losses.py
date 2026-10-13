"""RI discrepancy: token-level KL divergence over non-padded positions."""
import torch.nn.functional as F


def masked_kl_divergence(student_logits, teacher_logits, attention_mask):
    """KL(teacher || student), averaged over valid (non-padded) token positions.

    student_logits, teacher_logits: [batch, seq_len, vocab]
    attention_mask:                 [batch, seq_len], 1 for real tokens.
    """
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    student_logp = F.log_softmax(student_logits[..., :vocab].float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits[..., :vocab].float(), dim=-1)
    kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)  # [batch, seq_len]
    mask = attention_mask.float()
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)
