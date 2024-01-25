"""Script used to create the template graphs found in the template_graph folder."""

from utg import create_graph as cg
from utg import utils

if __name__ == "__main__":
    graphname = "concentric_small"
    G = cg.create_concentric_graph(radial=4, zones=2)
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )
    graphname = "concentric_large"
    G = cg.create_concentric_graph(radial=8, zones=5)
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )
    graphname = "cross"
    G = cg.create_radial_graph()
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )
    graphname = "star"
    G = cg.create_radial_graph(radial=8)
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )
    graphname = "block"
    G = cg.create_grid_graph()
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )
    graphname = "multiple_block"
    G = cg.create_grid_graph(rows=10, cols=10)
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )
    graphname = "manhattan"
    G = cg.create_grid_graph(rows=8, cols=8, width=50, height=300, diagonal=True)
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )
    graphname = "barcelona"
    G = cg.create_grid_graph(rows=10, cols=10, diagonal=True)
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )
    graphname = "fractal_cross"
    G = cg.create_fractal_graph(level=2)
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )
    graphname = "fractaler_cross"
    G = cg.create_fractal_graph(level=3)
    utils.save_graph(G, "./template_graph/" + graphname + ".graphml")
    utils.plot_graph(
        G, show=False, save=True, filepath="./template_graph/" + graphname + ".png"
    )
