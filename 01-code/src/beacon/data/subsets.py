"""Build the per-target-language evaluation subsets from three-hop evaluation verdicts.

  failure         (label=fail)    : a composition failure in target language X
  matched_success (label=success) : EN and X both succeed on two_hop

All subsets condition on EN two_hop success, so the fact is demonstrably stored in the model. The
failure set separates composition failures (the bridge is retrievable and the second hop works, but
the two-hop composition fails) from capability failures (the model cannot perform one of the hops).
A failure therefore requires, in X: two_hop INCORRECT, first_hop CORRECT (bridge B is retrievable),
AND second_hop CORRECT (B->C works given B). All three variants (two_hop, first_hop, second_hop) must
be evaluated upstream for this filter to apply.

The emitted item is the X-language two_hop item, i.e. the item read at the 'Answer:' colon.
"""
from __future__ import annotations

def _index(evaluated_items) -> dict:
    """(chain_id, lang, variant) -> item, for items that carry an eval verdict."""
    idx = {}
    for it in evaluated_items:
        if "eval" in it and "correct" in it["eval"]:
            idx[(it["chain_id"], it["lang"], it["variant"])] = it
    return idx

def build_subsets(evaluated_items, model_name: str, target_lang: str, routing_filter: bool = True) -> dict:
    """Return {'failure', 'matched_success', 'raw_failure_count'} from per-item eval flags.

    Conditioned on EN two_hop success. matched_success = X two_hop correct. failure = X two_hop wrong;
    with routing_filter (default), failure ALSO requires X first_hop AND X second_hop correct --- i.e.
    a genuine routing/composition failure, not a capability failure. raw_failure_count is the count
    BEFORE the routing filter (how many two_hop failures existed), which shows how many capability
    failures were removed. With routing_filter and no first/second-hop eval present, failures are
    dropped (they cannot be confirmed as composition failures), so all three hops must be evaluated.
    """
    idx = _index(evaluated_items)
    chain_ids = {cid for (cid, _l, _v) in idx}
    failure, matched_success, raw_fail = [], [], 0
    for cid in chain_ids:
        en = idx.get((cid, "en", "two_hop"))
        x = idx.get((cid, target_lang, "two_hop"))
        if en is None or x is None or not en["eval"]["correct"]:
            continue                                                                        
        if x["eval"]["correct"]:
            matched_success.append(x)
            continue
        raw_fail += 1                                                        
        if routing_filter:
            fh = idx.get((cid, target_lang, "first_hop"))
            sh = idx.get((cid, target_lang, "second_hop"))
            if fh is None or sh is None:
                continue                                                                                
            if not (fh["eval"]["correct"] and sh["eval"]["correct"]):
                continue                                                                   
        failure.append(x)
    return {"failure": failure, "matched_success": matched_success,
            "raw_failure_count": raw_fail, "routing_filtered": routing_filter}

def crosscheck_sizes(subsets, model_name, target_lang, configs, tol: float = 0.5) -> bool:
    """Compare subset sizes to configs/beacon.yaml expected sizes; warn outside the tolerance band.

    Subset sizes depend on the evaluated model outputs, so tol is a RELATIVE band (default 0.5 =
    within 50%), not equality. Returns True if both subsets land in band.
    """
    exp_fail = configs.beacon["expected_failure_sizes"].get(model_name, {}).get(target_lang)
    exp_succ = configs.beacon["expected_matched_success_sizes"].get(model_name, {}).get(target_lang)
    n_fail, n_succ = len(subsets["failure"]), len(subsets["matched_success"])

    def _in_band(n, exp):
        if exp is None or exp == 0:
            return True
        return abs(n - exp) / exp <= tol

    ok_fail, ok_succ = _in_band(n_fail, exp_fail), _in_band(n_succ, exp_succ)
    cell = f"{model_name}:{target_lang}"
    print(f"  [crosscheck] {cell}: failure {n_fail} (exp {exp_fail}) {'OK' if ok_fail else 'OUT'}; "
          f"success {n_succ} (exp {exp_succ}) {'OK' if ok_succ else 'OUT'}")
    if exp_fail is not None and exp_fail < 100:
        print(f"  [crosscheck] {cell}: expected failure < 100 --- a drop candidate (see beacon.yaml drop_cells)")
    return ok_fail and ok_succ
