def get_params_str(m): return f"{sum(p.numel() for p in m.parameters())/1e6:.1f}M"
