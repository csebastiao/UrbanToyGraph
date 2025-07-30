import networkx as nx
import numpy as np

import utg


class TestUtils:
    def test_osmnx(self):
        G = utg.make_osmnx_compatible(
            utg.create_grid_graph(rows=3, cols=3, multidigraph=False)
        )
        assert G.edges[0, 3, 0]["osmid"] == 0
        assert "crs" in G.graph.keys()
        assert G.graph["simplified"] is True
        assert isinstance(G, nx.MultiDiGraph)

    def test_true_zero(self):
        assert utg.utils.make_true_zero([1, 0.0000000001]) == [1, 0]

    def test_node_coord(self):
        assert utg.get_node_coord(
            utg.create_grid_graph(rows=3, cols=3, multidigraph=False, width=50), 1
        ) == [0, 50]

    def test_normalize(self):
        assert np.isclose(
            utg.utils.normalize([1, 1]), [1 / np.sqrt(2), 1 / np.sqrt(2)]
        ).all()

    def test_voronoi(self):
        points = [[-1, -1], [1, 1], [1, -1], [-1, 1]]
        bb = [-10, 10, -10, 10]
        vor = utg.utils.bounded_voronoi(points, bb)
        polygons = utg.utils.create_voronoi_polygons(vor, filtered=True)
        assert len(polygons) == 4
        # TODO add assertion once original function corrected
        # assert polygons[0].simplify(0) == shapely.Polygon([[-10, -10], [-10, 0], [0, 0], [0, -10]])
