"""Name -> bot factory. A bot has new_half(mem), choose_action(obs, mem), on_hand_end(res, mem)."""
import random
import types


def make_bot(name):
    if name == "agent" or name.startswith("agent|"):
        import ast
        from bot import core as agent
        kv = name.split("|", 1)[1].split(";") if "|" in name else []
        return agent.Agent({k: ast.literal_eval(v) for k, v in (x.split("=") for x in kv)})
    if name == "random":
        return _random_bot()
    from eval.zoo import ZOO
    return ZOO[name]()


def _random_bot():
    m = types.SimpleNamespace()
    m.new_half = lambda mem: None
    m.choose_action = lambda obs, mem: random.choice(obs.legal_actions)
    m.on_hand_end = lambda res, mem: None
    return m
