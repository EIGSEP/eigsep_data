"""Regression tests for eigsep_data.sim source-level behavior."""

import ast
from pathlib import Path


SIM_PATH = Path(__file__).resolve().parents[1] / "src" / "eigsep_data" / "sim.py"


def _gen_point_sources_node():
    module = ast.parse(SIM_PATH.read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "GlobalSim":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "gen_point_sources":
                    return method
    raise AssertionError("GlobalSim.gen_point_sources not found")


def test_gen_point_sources_achromatic_branch_uses_self_freqs():
    method = _gen_point_sources_node()
    if_node = next(node for node in method.body if isinstance(node, ast.If))
    achromatic_flux_expr = if_node.orelse[0].value

    bare_freq_names = [
        node.id
        for node in ast.walk(achromatic_flux_expr)
        if isinstance(node, ast.Name) and node.id == "freqs"
    ]
    assert not bare_freq_names

    self_freq_attrs = [
        node
        for node in ast.walk(achromatic_flux_expr)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == "freqs"
    ]
    assert self_freq_attrs


def test_gen_point_sources_achromatic_branch_keeps_frequency_broadcast():
    method = _gen_point_sources_node()
    if_node = next(node for node in method.body if isinstance(node, ast.If))
    achromatic_flux_expr = if_node.orelse[0].value

    pow_nodes = [node for node in ast.walk(achromatic_flux_expr) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)]
    assert any(isinstance(node.right, ast.Constant) and node.right.value == 0 for node in pow_nodes)
