"""Script used to create the template graphs found in the template_graph folder."""

from utg import create_graph as cg

if __name__ == "__main__":
    graphname = "concentric_small"
    G = cg.create_concentric_graph(radial=4, zones=2)
    cg.save_graph(G, "./template_graph/" + graphname + ".graphml")
    graphname = "concentric_large"
    G = cg.create_concentric_graph(radial=8, zones=5)
    cg.save_graph(G, "./template_graph/" + graphname + ".graphml")
    graphname = "cross"
    G = cg.create_radial_graph()
    cg.save_graph(G, "./template_graph/" + graphname + ".graphml")
    graphname = "star"
    G = cg.create_radial_graph(radial=8)
    cg.save_graph(G, "./template_graph/" + graphname + ".graphml")
    graphname = "block"
    G = cg.create_grid_graph()
    cg.save_graph(G, "./template_graph/" + graphname + ".graphml")
    graphname = "multiple_block"
    G = cg.create_grid_graph(rows=9, cols=9)
    cg.save_graph(G, "./template_graph/" + graphname + ".graphml")
    graphname = "manhattan"
    G = cg.create_grid_graph(rows=8, cols=8, width=50, height=300, diagonal=True)
    cg.save_graph(G, "./template_graph/" + graphname + ".graphml")
