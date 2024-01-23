"""Script used to create the template graphs found in the template_graph folder."""

from utg import create_graph as cg

if __name__ == "__main__":
    small_square = cg.create_grid_graph(m=3, n=3, width=50, height=100)
    print(small_square.edges[0, 3, 0]["length"])
