"""Nominal config metadata; never use it to slice heterogeneous attention layers."""


def nominal_dimensions(config, defaults):
    # Recent Transformers guards global per-layer attribute access. Serialization
    # preserves the nominal checkpoint values without disabling that model guard.
    values = config.to_dict()
    fields = {'n_layers': 'num_hidden_layers', 'n_heads': 'num_attention_heads',
              'n_kv_heads': 'num_key_value_heads', 'd_model': 'hidden_size',
              'head_dim': 'head_dim'}
    return {field: values.get(key, defaults[field]) for field, key in fields.items()}
