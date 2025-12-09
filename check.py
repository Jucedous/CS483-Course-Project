import pandas as pd

events = pd.read_csv("events_compact_all.csv", low_memory=False)
msgs   = pd.read_csv("messages_compact_all.csv", low_memory=False)

print(events["graph_num_edges"].describe())
print(events["graph_depth"].describe())
print(events["duration_hours"].describe())

print(msgs.head())
print(msgs["weak_role"].value_counts())
