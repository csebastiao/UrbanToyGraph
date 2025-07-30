import pytest
import networkx as nx
import math
import numpy as np
import shapely

import utg


class TestCreateGraph:
    @pytest.mark.filterwarnings("ignore: Height value selected")
    def test_grid(self):
        G_small = utg.create_grid_graph(
            rows=3, cols=5, width=50, height=None, multidigraph=False, diagonal=False
        )
        assert len(G_small) == 15
        assert len(G_small.edges) == 22
        assert isinstance(G_small, nx.Graph)
        assert G_small.edges[0, 1]["length"] == 50
        assert G_small.edges[0, 3]["length"] == 50
        G_big = utg.create_grid_graph(
            rows=3, cols=3, width=50, height=100, diagonal=True, multidigraph=True
        )
        assert len(G_big) == 9
        assert len(G_big.edges) == 28
        assert isinstance(G_big, nx.MultiDiGraph)
        assert G_big.edges[0, 1, 0]["length"] == 100
        assert G_big.edges[0, 3, 0]["length"] == 50
        assert math.isclose(G_big.edges[0, 4, 0]["length"], math.sqrt(50**2 + 100**2))

    @pytest.mark.filterwarnings("ignore: Height value selected")
    def test_distorted_grid(self):
        with pytest.raises(ValueError, match="Spacing needs to be between 0 and 0.99."):
            G = utg.create_distorted_grid_graph(spacing=12)
        G = utg.create_distorted_grid_graph(
            rows=3, cols=3, width=50, height=100, seed=7, spacing=0.95
        )
        rng = np.random.default_rng(7)
        assert len(G) == 9
        assert len(G.edges) == 12
        assert isinstance(G, nx.Graph)
        assert G.nodes[0]["x"] == rng.uniform(-0.95 * 25, 0.95 * 25)
        assert G.nodes[0]["y"] == rng.uniform(-0.95 * 50, 0.95 * 50)
        assert G.edges[0, 1]["geometry"] == shapely.LineString(
            [
                (G.nodes[0]["x"], G.nodes[0]["y"]),
                (G.nodes[1]["x"], G.nodes[1]["y"]),
            ]
        )
        assert isinstance(
            utg.create_distorted_grid_graph(multidigraph=True), nx.MultiDiGraph
        )

    def test_bridge(self):
        G_small = utg.create_bridge_graph(
            outrows=2,
            sscols=3,
            block_side=50,
            bridges=1,
            blength=150,
            multidigraph=False,
        )
        assert len(G_small) == 30
        assert len(G_small.edges) == 45
        assert isinstance(G_small, nx.Graph)
        assert G_small.edges[0, 1]["length"] == 50
        assert G_small.edges[12, 17]["length"] == 150
        G_big = utg.create_bridge_graph(
            outrows=2,
            sscols=3,
            block_side=50,
            bridges=2,
            blength=150,
            multidigraph=True,
        )
        assert len(G_big.edges) == 132
        assert isinstance(G_big, nx.MultiDiGraph)
        assert G_big.edges[16, 23, 0]["length"] == 150
        assert G_big.edges[18, 25, 0]["length"] == 150

    def test_radial(self):
        with pytest.raises(
            ValueError, match="Radial graph needs at least 3 radial roads to work."
        ):
            utg.create_radial_graph(2)
        G_small = utg.create_radial_graph(radial=6, length=100, multidigraph=False)
        assert len(G_small) == 7
        assert len(G_small.edges) == 6
        assert isinstance(G_small, nx.Graph)
        assert G_small.edges[0, 1]["length"] == 100
        assert math.isclose(G_small.nodes[2]["x"], 100 * np.cos(2 * np.pi / 6))
        assert math.isclose(G_small.nodes[2]["y"], 100 * np.sin(2 * np.pi / 6))
        G_big = utg.create_radial_graph(radial=6, length=100, multidigraph=True)
        assert len(G_big.edges) == 12
        assert isinstance(G_big, nx.MultiDiGraph)

    def test_concentric(self):
        with pytest.raises(
            ValueError,
            match="Concentric graph needs at least 2 radial positions to work.",
        ):
            utg.create_concentric_graph(radial=1)
        with pytest.raises(ValueError, match="Number of zones needs to be positive."):
            utg.create_concentric_graph(zones=0)
        G_small = utg.create_concentric_graph(
            radial=6,
            zones=2,
            radius=50,
            straight_edges=False,
            center=False,
            multidigraph=False,
        )
        assert len(G_small) == 12
        assert len(G_small.edges) == 18
        assert isinstance(G_small, nx.Graph)
        # TODO test curved linestring geometry
        assert math.isclose(G_small.nodes[1]["x"], 50 * np.cos(2 * np.pi / 6))
        assert math.isclose(G_small.nodes[1]["y"], 50 * np.sin(2 * np.pi / 6))
        G_big = utg.create_concentric_graph(
            radial=4,
            zones=2,
            radius=50,
            straight_edges=True,
            center=True,
            multidigraph=True,
        )
        assert len(G_big) == 9
        assert len(G_big.edges) == 32
        assert isinstance(G_big, nx.MultiDiGraph)
        assert G_big.nodes[0]["x"] == 0
        assert G_big.nodes[0]["y"] == 0
        assert G_big.edges[0, 1, 0]["length"] == 50
        assert math.isclose(G_big.edges[1, 2, 0]["length"], math.sqrt(50**2 + 50**2))

    def test_fractal(self):
        with pytest.raises(ValueError, match="Level needs to be superior to 2."):
            utg.create_fractal_graph(level=1)
        G_small = utg.create_fractal_graph(
            branch=6, level=2, inital_length=100, multidigraph=False
        )
        assert len(G_small) == 37
        assert len(G_small.edges) == 36
        assert isinstance(G_small, nx.Graph)
        assert G_small.edges[0, 1]["length"] == 100
        # TODO test length smaller fractal edge
        # TODO test position smaller fractal node
        G_big = utg.create_fractal_graph(
            branch=4, level=3, inital_length=100, multidigraph=True
        )
        assert len(G_big) == 53
        assert len(G_big.edges) == 104
        assert isinstance(G_big, nx.MultiDiGraph)

    def test_add_random(self):
        G_small = utg.create_grid_graph(
            rows=3, cols=3, width=50, height=None, multidigraph=False, diagonal=False
        )
        G_small = utg.add_random_edges(G_small, N=1)
        assert len(G_small) == 9
        assert len(G_small.edges) == 13
        assert isinstance(G_small, nx.Graph)
        # TODO test with seed once added if right edge is added
        G_big = utg.create_grid_graph(
            rows=3, cols=3, width=50, height=None, multidigraph=True, diagonal=False
        )
        G_big = utg.add_random_edges(G_big, N=1)
        assert len(G_big) == 9
        assert len(G_big.edges) == 26
        assert isinstance(G_big, nx.MultiDiGraph)
        # TODO test error from number of edges added more than maximal planar graph
        with pytest.warns(
            UserWarning,
            match="1000 consecutive random trials without finding an edge to add, verify that there are edges that can be added before retrying.",
        ):
            utg.add_random_edges(G_big, N=50)

    def test_remove_random(self):
        G_small = utg.create_grid_graph(
            rows=3, cols=3, width=50, height=None, multidigraph=False, diagonal=False
        )
        with pytest.raises(
            ValueError, match="N is too large for the graph, pick a smaller N"
        ):
            utg.remove_random_edges(
                G_small, keep_all_nodes=True, prevent_disconnect=True, N=5
            )
        with pytest.raises(
            ValueError, match="N is too large for the graph, pick a smaller N"
        ):
            utg.remove_random_edges(
                G_small, keep_all_nodes=True, prevent_disconnect=False, N=8
            )
        with pytest.raises(
            ValueError, match="N is too large for the graph, pick a smaller N"
        ):
            utg.remove_random_edges(
                G_small, keep_all_nodes=False, prevent_disconnect=False, N=12
            )
        G_small_one = utg.remove_random_edges(
            G_small, N=1, keep_all_nodes=True, prevent_disconnect=True
        )
        assert len(G_small_one) == 9
        assert len(G_small_one.edges) == 11
        assert isinstance(G_small_one, nx.Graph)
        assert (
            nx.number_connected_components(
                utg.remove_random_edges(
                    G_small, N=8, keep_all_nodes=False, prevent_disconnect=False
                )
            )
            > 1
        )
        # TODO test with seed once added if right node is removed
        G_big = utg.create_grid_graph(
            rows=3, cols=3, width=50, height=None, multidigraph=True, diagonal=False
        )
        G_big_one = utg.remove_random_edges(
            G_big, N=1, keep_all_nodes=True, prevent_disconnect=True
        )
        assert len(G_big_one) == 9
        assert len(G_big_one.edges) == 22
        assert isinstance(G_big_one, nx.MultiDiGraph)
