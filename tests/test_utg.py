from utg.create_graph import (
    create_grid_graph,
    create_radial_graph,
    create_concentric_graph,
    remove_random_edges,
)


def test_grid():
    assert len(create_grid_graph(m=3, n=3)) == 9
    assert len(create_grid_graph(m=3, n=5)) == 15
    assert len(create_grid_graph(m=3, n=3, multidigraph=True).edges) == 24
    assert len(create_grid_graph(m=3, n=3, multidigraph=False).edges) == 12
    assert (
        create_grid_graph(width=50).edges[0, 1, 0]["length"] == 50
        and create_grid_graph(width=50).edges[0, 3, 0]["length"] == 50
    )
    assert (
        create_grid_graph(width=50, height=100).edges[0, 1, 0]["length"] == 100
        and create_grid_graph(width=50, height=100).edges[0, 3, 0]["length"] == 50
    )


def test_radial():
    assert len(create_radial_graph(radial=4)) == 5
    assert len(create_radial_graph(radial=4, multidigraph=True).edges) == 8
    assert len(create_radial_graph(radial=4, multidigraph=False).edges) == 4
    assert create_radial_graph(length=100).edges[0, 1, 0]["length"] == 100


def test_concentric():
    assert len(create_concentric_graph(radial=4, zones=2, center=True)) == 9
    assert len(create_concentric_graph(radial=4, zones=2, center=False)) == 8
    assert len(create_concentric_graph(radial=8, zones=5, center=True)) == 41
    assert (
        len(
            create_concentric_graph(
                radial=4, zones=2, center=True, multidigraph=True
            ).edges
        )
        == 32
    )
    assert (
        len(
            create_concentric_graph(
                radial=4, zones=2, center=True, multidigraph=False
            ).edges
        )
        == 16
    )
    assert (
        len(
            create_concentric_graph(
                radial=4, zones=2, center=False, multidigraph=True
            ).edges
        )
        == 24
    )
    assert (
        len(
            create_concentric_graph(
                radial=4, zones=2, center=False, multidigraph=False
            ).edges
        )
        == 12
    )
    assert (
        create_concentric_graph(radial=4, radius=50, center=True).edges[0, 1, 0][
            "length"
        ]
        == 50
    )
    assert (
        create_concentric_graph(radial=4, radius=50, center=True).edges[1, 5, 0][
            "length"
        ]
        == 50
    )


def test_remove_edge():
    for func in [create_grid_graph, create_radial_graph, create_concentric_graph]:
        G = func(multidigraph=True)
        init_len = len(G.edges)
        G = remove_random_edges(G, N=1, keep_all_nodes=False, prevent_disconnect=True)
        assert init_len == len(G.edges) + 2
        G = func(multidigraph=False)
        init_len = len(G.edges)
        G = remove_random_edges(
            G, N=1, keep_all_nodes=False, prevent_disconnect=True, is_directed=False
        )
        assert init_len == len(G.edges) + 1