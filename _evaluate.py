from collections import defaultdict

def evaluate(preds, real, metrics, results, device):

    '''Compute evaluation metrics'''
    # Transfer tensors to other device to avoid issues with memory leak
    other_device = 'cuda:1' if device == 'cuda:0' else 'cuda:0'
    m_preds = preds.to(other_device).detach()
    m_real = real.to(other_device).detach()

    # Compute metrics
    raw_metrics = metrics(m_preds, m_real)
    for k, v in raw_metrics.items():
        results[k].append(v.item())

    print('metrics shape:', f'({len(raw_metrics)}, {len(raw_metrics["ssim"])})')

    return results