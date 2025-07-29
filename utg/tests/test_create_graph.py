import pytest

import utg


class TestCreateGraph:
    def test_grid(self):
        pass

    def test_distorted_grid(self):
        pass

    def test_bridge(self):
        pass

    def test_radial(self):
        pass

    def test_concentric(self):
        pass

    def test_curved(self):
        pass

    def test_fractal(self):
        pass

    def test_add_random(self):
        pass

    def test_remove_random(self):
        pass


# TODO: Put into good structure and fill out
# TODO: Test raised warnings and errors
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_grid():
    assert len(utg.create_grid_graph(rows=3, cols=3)) == 9
    assert len(utg.create_grid_graph(rows=3, cols=5)) == 15
    assert len(utg.create_grid_graph(rows=3, cols=3, multidigraph=True).edges) == 24
    assert len(utg.create_grid_graph(rows=3, cols=3, multidigraph=False).edges) == 12
    assert (
        utg.create_grid_graph(width=50).edges[0, 1]["length"] == 50
        and utg.create_grid_graph(width=50).edges[0, 3]["length"] == 50
    )
    assert (
        utg.create_grid_graph(width=50, height=100).edges[0, 1]["length"] == 100
        and utg.create_grid_graph(width=50, height=100).edges[0, 3]["length"] == 50
    )


def test_bridge():
    assert len(utg.create_bridge_graph(sscols=3, outrows=2, bridges=1)) == 30
    assert (
        len(
            utg.create_bridge_graph(
                sscols=3, outrows=2, bridges=1, multidigraph=True
            ).edges
        )
        == 90
    )
    assert (
        len(
            utg.create_bridge_graph(
                sscols=3, outrows=2, bridges=1, multidigraph=False
            ).edges
        )
        == 45
    )
    assert (
        utg.create_bridge_graph(sscols=3, outrows=2, bridges=1, blength=100).edges[
            0, 1
        ]["length"]
        == 50
    )
    assert (
        utg.create_bridge_graph(sscols=3, outrows=2, bridges=1, blength=100).edges[
            12, 17
        ]["length"]
        == 100
    )


def test_radial():
    assert len(utg.create_radial_graph(radial=4)) == 5
    assert len(utg.create_radial_graph(radial=4, multidigraph=True).edges) == 8
    assert len(utg.create_radial_graph(radial=4, multidigraph=False).edges) == 4
    assert utg.create_radial_graph(length=100).edges[0, 1]["length"] == 100


def test_concentric():
    assert len(utg.create_concentric_graph(radial=4, zones=2, center=True)) == 9
    assert len(utg.create_concentric_graph(radial=4, zones=2, center=False)) == 8
    assert len(utg.create_concentric_graph(radial=8, zones=5, center=True)) == 41
    assert (
        len(
            utg.create_concentric_graph(
                radial=4, zones=2, center=True, multidigraph=True
            ).edges
        )
        == 32
    )
    assert (
        len(
            utg.create_concentric_graph(
                radial=4, zones=2, center=True, multidigraph=False
            ).edges
        )
        == 16
    )
    assert (
        len(
            utg.create_concentric_graph(
                radial=4, zones=2, center=False, multidigraph=True
            ).edges
        )
        == 24
    )
    assert (
        len(
            utg.create_concentric_graph(
                radial=4, zones=2, center=False, multidigraph=False
            ).edges
        )
        == 12
    )
    assert (
        utg.create_concentric_graph(radial=4, radius=50, center=True).edges[0, 1][
            "length"
        ]
        == 50
    )
    assert (
        utg.create_concentric_graph(radial=4, radius=50, center=True).edges[1, 5][
            "length"
        ]
        == 50
    )


def test_fractal():
    assert len(utg.create_fractal_graph(branch=4, level=2)) == 17
    assert len(utg.create_fractal_graph(branch=4, level=3)) == 53
    assert (
        len(utg.create_fractal_graph(branch=4, level=2, multidigraph=False).edges) == 16
    )
    assert (
        len(utg.create_fractal_graph(branch=4, level=2, multidigraph=True).edges) == 32
    )


def test_remove_edge():
    for func in [
        utg.create_grid_graph,
        utg.create_bridge_graph,
        utg.create_radial_graph,
        utg.create_concentric_graph,
        utg.create_fractal_graph,
    ]:
        G = func(multidigraph=True)
        init_len = len(G.edges)
        G = utg.remove_random_edges(
            G, N=1, keep_all_nodes=False, prevent_disconnect=True
        )
        assert init_len == len(G.edges) + 2
        G = func(multidigraph=False)
        init_len = len(G.edges)
        G = utg.remove_random_edges(
            G, N=1, keep_all_nodes=False, prevent_disconnect=True, is_directed=False
        )
        assert init_len == len(G.edges) + 1


def test_add_edge():
    for func in [
        utg.create_grid_graph,
        utg.create_bridge_graph,
        utg.create_radial_graph,
        utg.create_concentric_graph,
        utg.create_fractal_graph,
    ]:
        # G = func(multidigraph=True)
        # init_len = len(G.edges)
        # G = add_random_edges(G, N=1, is_directed=True)
        # assert init_len == len(G.edges) - 2
        G = func(multidigraph=False)
        init_len = len(G.edges)
        G = utg.add_random_edges(G, N=1, is_directed=False)
        assert init_len == len(G.edges) - 1
