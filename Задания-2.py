import networkx as nx
import matplotlib.pyplot as plt
                     ##Members of your group (Only for centrality evaluation)
group_members = ['Alice', 'Boris', 'Victor', 'Galina']
                     ##Friends+friends of friends graph(example VK style dataset)
                     #Replace with real VK data if needed
edges = [
    ('Alice', 'Boris'),
    ('Alice', 'Victor'),
    ('Boris', 'Galina'),
    ('Victor', 'Galina'),
    ('Alice', 'Olga'),
    ('Olga', 'Pavel'),
    ('Pavel', 'Galina'),
    ('Boris', 'Dmitry'),
    ('Dmitry', 'Elena'),
    ('Elena', 'Victor'),
]
G = nx.Graph()
G.add_edges_from(edges)
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(G)

print("Centrality scores for GROUP MEMBERS ONLY:\n")
print("{:<10} {:<15} {:<15} {:<15}".format("Name", "Betweenness", "Closeness", "Eigenvector"))

for member in group_members:
    print("{:<10} {:<15.4f} {:<15.4f} {:<15.4f}".format(
        member,
        betweenness.get(member, 0),
        closeness.get(member, 0),
        eigenvector.get(member, 0)
    ))

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)

nx.draw( G, pos, with_labels=True,
    node_color='lightblue', node_size=1200,
    edge_color='gray', font_size=10
)
plt.title("VK Friends & Friends of Friends Network")
plt.show()