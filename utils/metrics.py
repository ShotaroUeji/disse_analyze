# utils/metrics.py
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None


@torch.no_grad()
def cosine_sim(q_emb: torch.Tensor, g_emb: torch.Tensor) -> torch.Tensor:
    """
    Pairwise cosine similarity.

    Args:
        q_emb: [N_query, D]
        g_emb: [N_gallery, D]

    Returns:
        S: [N_query, N_gallery]
    """
    q = F.normalize(q_emb, dim=-1)
    g = F.normalize(g_emb, dim=-1)
    return q @ g.T


@torch.no_grad()
def recall_at_k_multi(
    S: torch.Tensor,
    query_ids: torch.Tensor,
    gallery_ids: torch.Tensor,
    ks=(1, 5, 10),
    mask_self: bool = False,
):
    """
    Multi-positive Recall@K.

    Args:
        S: [N_query, N_gallery] similarity matrix.
        query_ids: [N_query] label IDs for queries.
        gallery_ids: [N_gallery] label IDs for gallery items.
        ks: K values.
        mask_self: True for intra-modal retrieval where query and gallery
                   are the same set. It masks the diagonal.

    Returns:
        dict with R@K, MedR, MnR.
    """
    query_ids = query_ids.to(S.device)
    gallery_ids = gallery_ids.to(S.device)

    S = S.clone()

    if mask_self:
        if S.size(0) != S.size(1):
            raise ValueError("mask_self=True requires a square similarity matrix.")
        S.fill_diagonal_(-float("inf"))

    same = query_ids[:, None].eq(gallery_ids[None, :])  # [Nq, Ng]

    if mask_self:
        same.fill_diagonal_(False)

    ranks = S.argsort(dim=1, descending=True)  # [Nq, Ng]

    out = {}
    for k in ks:
        kk = min(int(k), S.size(1))
        if kk <= 0:
            out[f"R@{k}"] = float("nan")
            continue
        topk = ranks[:, :kk]
        hit = same.gather(1, topk).any(dim=1).float()
        out[f"R@{k}"] = hit.mean().item()

    first = torch.full(
        (S.size(0),),
        S.size(1) + 1,
        dtype=torch.long,
        device=S.device,
    )

    for i in range(S.size(0)):
        row = same[i][ranks[i]]
        pos = torch.nonzero(row, as_tuple=False)
        if pos.numel() > 0:
            first[i] = pos[0, 0] + 1

    valid = first <= S.size(1)
    if valid.any():
        out["MedR"] = first[valid].median().item()
        out["MnR"] = first[valid].float().mean().item()
    else:
        out["MedR"] = float("nan")
        out["MnR"] = float("nan")

    return out


@torch.no_grad()
def recall_at_k(
    sim: torch.Tensor,
    ids_t: torch.Tensor,
    ids_a: torch.Tensor,
    ks=(1, 5, 10),
):
    """
    Backward-compatible wrapper for single/multi-positive retrieval.
    """
    return recall_at_k_multi(sim, ids_t, ids_a, ks=ks, mask_self=False)


@torch.no_grad()
def chance_at_k_multi(
    query_ids: torch.Tensor,
    gallery_ids: torch.Tensor,
    ks=(1, 5, 10),
):
    """
    Theoretical random-hit probability for multi-positive Recall@K.

    For each query with P positives in a gallery of size Ng:
        Chance@K = 1 - C(Ng - P, K) / C(Ng, K)

    This is not part of the original paper tables, but useful as an
    additional diagnostic.
    """
    device = gallery_ids.device
    query_ids = query_ids.to(device)
    gallery_ids = gallery_ids.to(device)

    Ng = int(gallery_ids.numel())
    if Ng == 0:
        return {f"Chance@{k}": float("nan") for k in ks}

    P = torch.stack([(gallery_ids == q).sum() for q in query_ids]).float().to(device)
    Ng_t = torch.tensor(float(Ng), device=device)

    out = {}
    for k in ks:
        kk = min(int(k), Ng)
        if kk <= 0:
            out[f"Chance@{k}"] = 0.0
            continue

        k_t = torch.tensor(float(kk), device=device)

        # no_hit = C(Ng-P, kk) / C(Ng, kk)
        # Use lgamma for numerical stability.
        valid = (Ng_t - P) >= k_t

        no_hit = torch.zeros_like(P)
        no_hit[valid] = torch.exp(
            torch.lgamma(Ng_t - P[valid] + 1)
            - torch.lgamma(Ng_t - P[valid] - k_t + 1)
            - torch.lgamma(Ng_t + 1)
            + torch.lgamma(Ng_t - k_t + 1)
        )

        chance = (1.0 - no_hit).clamp(0.0, 1.0)
        out[f"Chance@{k}"] = chance.mean().item()

    return out


def _make_pair_ids(src_ids: torch.Tensor, spa_ids: torch.Tensor) -> torch.Tensor:
    """
    Make unique IDs for (source, space) pairs.
    Works even when src_ids/spa_ids are not contiguous.
    """
    src_ids_cpu = src_ids.detach().cpu().tolist()
    spa_ids_cpu = spa_ids.detach().cpu().tolist()

    mapping = {}
    pair_ids = []
    next_id = 0

    for s, p in zip(src_ids_cpu, spa_ids_cpu):
        key = (int(s), int(p))
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        pair_ids.append(mapping[key])

    return torch.tensor(pair_ids, dtype=torch.long, device=src_ids.device)


@torch.no_grad()
def _eval_block(
    prefix: str,
    q_emb: torch.Tensor,
    g_emb: torch.Tensor,
    q_ids: torch.Tensor,
    g_ids: torch.Tensor,
    ks=(1, 5, 10),
    mask_self: bool = False,
    add_chance: bool = True,
    q2g_name: str = "Q2G",
    g2q_name: str = "G2Q",
):
    """
    Generic retrieval block.

    q_emb -> g_emb and g_emb -> q_emb are both evaluated.
    """
    S = cosine_sim(q_emb, g_emb)  # [Nq, Ng]

    R_q2g = recall_at_k_multi(S, q_ids, g_ids, ks=ks, mask_self=mask_self)
    R_g2q = recall_at_k_multi(S.T, g_ids, q_ids, ks=ks, mask_self=mask_self)

    mets = {f"{prefix}/{q2g_name}/{k}": v for k, v in R_q2g.items()}
    mets |= {f"{prefix}/{g2q_name}/{k}": v for k, v in R_g2q.items()}

    if add_chance:
        C_q2g = chance_at_k_multi(q_ids, g_ids, ks=ks)
        C_g2q = chance_at_k_multi(g_ids, q_ids, ks=ks)

        mets |= {f"{prefix}/{q2g_name}/{k}": v for k, v in C_q2g.items()}
        mets |= {f"{prefix}/{g2q_name}/{k}": v for k, v in C_g2q.items()}

        for k in ks:
            mets[f"{prefix}/{q2g_name}/Excess@{k}"] = (
                R_q2g[f"R@{k}"] - C_q2g[f"Chance@{k}"]
            )
            mets[f"{prefix}/{g2q_name}/Excess@{k}"] = (
                R_g2q[f"R@{k}"] - C_g2q[f"Chance@{k}"]
            )

    return mets


@torch.no_grad()
def eval_retrieval(
    a_spa: torch.Tensor,
    a_src: torch.Tensor,
    t_spa: torch.Tensor,
    t_src: torch.Tensor,
    src_lb: torch.Tensor,
    spa_lb: torch.Tensor,
    device=None,
    use_wandb: bool = True,
    epoch=None,
    ks=(1, 5, 10),
    add_chance: bool = True,
    eval_intra: bool = True,
    eval_cross: bool = True,
    eval_both: bool = True,
):
    """
    Retrieval evaluation for DISSE-style source/spatial disentangled embeddings.

    Paper correspondence:
      - On-task source retrieval:
          source embedding, correctness = source ID
      - On-task spatial retrieval:
          spatial embedding, correctness = space ID
      - Off-task source retrieval:
          spatial embedding, correctness = source ID
      - Off-task spatial retrieval:
          source embedding, correctness = space ID
      - Both-task retrieval:
          concat(source embedding, spatial embedding), correctness = (source ID, space ID)

    Evaluated directions:
      - Cross-modal: txt -> aud, aud -> txt
      - Intra-modal: aud -> aud, txt -> txt
        Intra-modal retrieval masks the diagonal item.
    """
    if device is not None:
        a_spa = a_spa.to(device)
        a_src = a_src.to(device)
        t_spa = t_spa.to(device)
        t_src = t_src.to(device)
        src_lb = src_lb.to(device)
        spa_lb = spa_lb.to(device)

    ids_src = src_lb
    ids_spa = spa_lb
    ids_both = _make_pair_ids(src_lb, spa_lb)

    mets = {}

    # ------------------------------------------------------------------
    # Cross-modal retrieval: Table 4 in the paper
    # ------------------------------------------------------------------
    if eval_cross:
        # On-task
        mets |= _eval_block(
            "Cross/SRC",
            t_src,
            a_src,
            ids_src,
            ids_src,
            ks=ks,
            mask_self=False,
            add_chance=add_chance,
            q2g_name="T2A",
            g2q_name="A2T",
        )
        mets |= _eval_block(
            "Cross/SPA",
            t_spa,
            a_spa,
            ids_spa,
            ids_spa,
            ks=ks,
            mask_self=False,
            add_chance=add_chance,
            q2g_name="T2A",
            g2q_name="A2T",
        )

        # Off-task
        # X-SRC: use spatial embeddings, judge by source labels.
        mets |= _eval_block(
            "Cross/X-SRC",
            t_spa,
            a_spa,
            ids_src,
            ids_src,
            ks=ks,
            mask_self=False,
            add_chance=add_chance,
            q2g_name="T2A",
            g2q_name="A2T",
        )

        # X-SPA: use source embeddings, judge by spatial labels.
        mets |= _eval_block(
            "Cross/X-SPA",
            t_src,
            a_src,
            ids_spa,
            ids_spa,
            ks=ks,
            mask_self=False,
            add_chance=add_chance,
            q2g_name="T2A",
            g2q_name="A2T",
        )

        if eval_both:
            t_both = torch.cat([t_src, t_spa], dim=-1)
            a_both = torch.cat([a_src, a_spa], dim=-1)

            mets |= _eval_block(
                "Cross/BOTH",
                t_both,
                a_both,
                ids_both,
                ids_both,
                ks=ks,
                mask_self=False,
                add_chance=add_chance,
                q2g_name="T2A",
                g2q_name="A2T",
            )

    # ------------------------------------------------------------------
    # Intra-modal retrieval: Table 3 in the paper
    # ------------------------------------------------------------------
    if eval_intra:
        # Audio -> Audio, diagonal masked
        mets |= _eval_block(
            "IntraAudio/SRC",
            a_src,
            a_src,
            ids_src,
            ids_src,
            ks=ks,
            mask_self=True,
            add_chance=add_chance,
            q2g_name="A2A",
            g2q_name="A2A_T",
        )
        mets |= _eval_block(
            "IntraAudio/SPA",
            a_spa,
            a_spa,
            ids_spa,
            ids_spa,
            ks=ks,
            mask_self=True,
            add_chance=add_chance,
            q2g_name="A2A",
            g2q_name="A2A_T",
        )

        # Off-task intra-audio
        mets |= _eval_block(
            "IntraAudio/X-SRC",
            a_spa,
            a_spa,
            ids_src,
            ids_src,
            ks=ks,
            mask_self=True,
            add_chance=add_chance,
            q2g_name="A2A",
            g2q_name="A2A_T",
        )
        mets |= _eval_block(
            "IntraAudio/X-SPA",
            a_src,
            a_src,
            ids_spa,
            ids_spa,
            ks=ks,
            mask_self=True,
            add_chance=add_chance,
            q2g_name="A2A",
            g2q_name="A2A_T",
        )

        # Text -> Text, diagonal masked
        mets |= _eval_block(
            "IntraText/SRC",
            t_src,
            t_src,
            ids_src,
            ids_src,
            ks=ks,
            mask_self=True,
            add_chance=add_chance,
            q2g_name="T2T",
            g2q_name="T2T_T",
        )
        mets |= _eval_block(
            "IntraText/SPA",
            t_spa,
            t_spa,
            ids_spa,
            ids_spa,
            ks=ks,
            mask_self=True,
            add_chance=add_chance,
            q2g_name="T2T",
            g2q_name="T2T_T",
        )

        # Off-task intra-text
        mets |= _eval_block(
            "IntraText/X-SRC",
            t_spa,
            t_spa,
            ids_src,
            ids_src,
            ks=ks,
            mask_self=True,
            add_chance=add_chance,
            q2g_name="T2T",
            g2q_name="T2T_T",
        )
        mets |= _eval_block(
            "IntraText/X-SPA",
            t_src,
            t_src,
            ids_spa,
            ids_spa,
            ks=ks,
            mask_self=True,
            add_chance=add_chance,
            q2g_name="T2T",
            g2q_name="T2T_T",
        )

    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed, but use_wandb=True.")
        wandb.log({"epoch": epoch, **mets})

    return mets


# ----------------------------------------------------------------------
# Pairwise cosine distance
# ----------------------------------------------------------------------
def _pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=1)
    return 1.0 - (x @ x.T)


# ----------------------------------------------------------------------
# Invariance Ratio / IIDR-like diagnostic
# ----------------------------------------------------------------------
@torch.no_grad()
def invariance_ratio(
    emb: torch.Tensor,
    src_ids: torch.Tensor,
    spa_ids: torch.Tensor,
) -> dict:
    """
    Invariance ratio diagnostic.

    Args:
        emb: [N, D], e.g. audio_source, audio_space, text_source, text_space.
        src_ids: [N] source IDs.
        spa_ids: [N] spatial IDs.

    Definitions:
        same_space_diff_source:
            pairs with same space and different source
        same_source_diff_space:
            pairs with same source and different space

    Interpretation:
        For a spatial embedding:
            IR_space = mean distance(same source, diff space)
                     / mean distance(same space, diff source)
            Larger is better.

        For a source embedding:
            IR_source = mean distance(same space, diff source)
                      / mean distance(same source, diff space)
            Larger is better.
    """
    src_ids = src_ids.to(emb.device)
    spa_ids = spa_ids.to(emb.device)

    N = emb.size(0)
    if N < 2:
        return {
            "IR_space": float("nan"),
            "IR_source": float("nan"),
            "num_ss_ds": 0,
            "num_sd_ss": 0,
            "d_ss_ds": float("nan"),
            "d_sd_ss": float("nan"),
        }

    dist = _pairwise_cosine_distance(emb)

    eye = torch.eye(N, dtype=torch.bool, device=emb.device)
    src_eq = src_ids[:, None].eq(src_ids[None, :])
    spa_eq = spa_ids[:, None].eq(spa_ids[None, :])
    valid = ~eye

    same_space_diff_source = spa_eq & (~src_eq) & valid
    same_source_diff_space = src_eq & (~spa_eq) & valid

    d_ss_ds = dist[same_space_diff_source]
    d_sd_ss = dist[same_source_diff_space]

    eps = 1e-8

    if d_ss_ds.numel() > 0 and d_sd_ss.numel() > 0:
        d_ss_ds_mean = d_ss_ds.mean()
        d_sd_ss_mean = d_sd_ss.mean()

        IR_space = (d_sd_ss_mean / (d_ss_ds_mean + eps)).item()
        IR_source = (d_ss_ds_mean / (d_sd_ss_mean + eps)).item()

        d_ss_ds_value = d_ss_ds_mean.item()
        d_sd_ss_value = d_sd_ss_mean.item()
    else:
        IR_space = float("nan")
        IR_source = float("nan")
        d_ss_ds_value = float("nan")
        d_sd_ss_value = float("nan")

    return {
        "IR_space": IR_space,
        "IR_source": IR_source,
        "num_ss_ds": int(same_space_diff_source.sum().item()),
        "num_sd_ss": int(same_source_diff_space.sum().item()),
        "d_ss_ds": d_ss_ds_value,
        "d_sd_ss": d_sd_ss_value,
    }


@torch.no_grad()
def eval_invariance_all(
    a_spa: torch.Tensor,
    a_src: torch.Tensor,
    t_spa: torch.Tensor,
    t_src: torch.Tensor,
    src_lb: torch.Tensor,
    spa_lb: torch.Tensor,
    device=None,
    use_wandb: bool = True,
    epoch=None,
):
    """
    Convenience wrapper to log invariance_ratio for all four embeddings.
    """
    if device is not None:
        a_spa = a_spa.to(device)
        a_src = a_src.to(device)
        t_spa = t_spa.to(device)
        t_src = t_src.to(device)
        src_lb = src_lb.to(device)
        spa_lb = spa_lb.to(device)

    vals = {
        "IR/audio_spatial": invariance_ratio(a_spa, src_lb, spa_lb),
        "IR/audio_source": invariance_ratio(a_src, src_lb, spa_lb),
        "IR/text_spatial": invariance_ratio(t_spa, src_lb, spa_lb),
        "IR/text_source": invariance_ratio(t_src, src_lb, spa_lb),
    }

    flat = {}
    for prefix, d in vals.items():
        for k, v in d.items():
            flat[f"{prefix}/{k}"] = v

    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed, but use_wandb=True.")
        wandb.log({"epoch": epoch, **flat})

    return flat
