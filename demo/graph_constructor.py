"""Script used to create the template graphs found in the template_graph folder."""

import utg

if __name__ == "__main__":
    graphname = "concentric_small"
    G = utg.create_concentric_graph(radial=4, zones=2)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "concentric_large"
    G = utg.create_concentric_graph(radial=8, zones=5)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "concentric_straight"
    G = utg.create_concentric_graph(radial=4, zones=2, straight_edges=True)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "cross"
    G = utg.create_radial_graph()
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "star"
    G = utg.create_radial_graph(radial=8)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "block"
    G = utg.create_grid_graph()
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "multiple_block"
    G = utg.create_grid_graph(rows=10, cols=10)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "manhattan"
    G = utg.create_grid_graph(rows=8, cols=8, width=50, height=300, diagonal=True)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "barcelona"
    G = utg.create_grid_graph(rows=10, cols=10, diagonal=True)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "bridge_small"
    G = utg.create_bridge_graph()
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "bridge_large"
    G = utg.create_bridge_graph(outrows=4, sscols=4, bridges=2)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "fractal_cross"
    G = utg.create_fractal_graph(level=2)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "fractaler_cross"
    G = utg.create_fractal_graph(level=3)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
    graphname = "fractalerer_cross"
    G = utg.create_fractal_graph(level=4)
    utg.save_graph(G, "./demo/template_graph/" + graphname + ".graphml")
    utg.plot_graph(
        G,
        show=False,
        save=True,
        close=True,
        filepath="./demo/template_graph/" + graphname + ".png",
    )
