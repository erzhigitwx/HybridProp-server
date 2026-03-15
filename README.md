# HybridProp RecSys

Multimodal restaurant recommendation system with **soft-boundary filtering**.

Built on the Yelp Open Dataset: 200K photos, 8.6M reviews, 160K+ businesses.

## What makes this different

Traditional filters are binary — set "4+ stars" and a 3.9-star restaurant that perfectly matches your taste disappears. HybridProp uses **soft filters**: when strict results are scarce, it relaxes boundaries and resurfaces items whose visual/text profile closely matches your preferences, even if they're slightly outside your specified range. Each result includes a `filter_match` score (0.0–1.0) so the frontend can show *"slightly below your filter, but matches your taste"*.

## Architecture

```
Photos (CLIP ViT-B/32) ─┐
Reviews (MiniLM-L6)     ─┼→ Restaurant Tower (MLP) ─→ 256d embedding ─→ Qdrant
Tabular features        ─┘

User interaction history ─→ User Tower (attention) ─→ 256d query ─→ Soft Filter Search
```

**Two-tower model** with contrastive learning (BPR/InfoNCE):
- **Restaurant tower**: projects concatenated CLIP + text + tabular features to shared 256d space
- **User tower**: multi-head attention over interaction history weighted by rating and recency
- **Soft filter**: 2-phase Qdrant search (strict → relaxed → rerank with filter bonus)

### Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- GPU with 8GB+ VRAM (RTX 3070 tested) - or CPU (slower)
- Yelp Open Dataset (~12 GB): https://www.yelp.com/dataset (1GB active)