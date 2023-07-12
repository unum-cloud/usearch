"""
rm -rf datasets/cc_3M/images.usearch datasets/cc_3M/texts.usearch
python python/scripts/join.py
"""
from numpy import dot
from numpy.linalg import norm


from usearch.index import Index, MetricKind
from usearch.io import load_matrix
from usearch.eval import measure_seconds, recall_members

a_mat = load_matrix("datasets/cc_3M/texts.fbin", view=True)
b_mat = load_matrix("datasets/cc_3M/images.fbin", view=True)

print(f"Loaded two datasets of shape: {a_mat.shape}, {b_mat.shape}")

a = Index(
    a_mat.shape[1],
    MetricKind.Cos,
    path="datasets/cc_3M/texts.usearch",
    dtype="f32",
)
b = Index(
    b_mat.shape[1],
    MetricKind.Cos,
    path="datasets/cc_3M/images.usearch",
    dtype="f32",
)

if len(a) == 0:
    a.add(None, a_mat, log=True, batch_size=1024 * 32)
    a.save()

if len(b) == 0:
    b.add(None, b_mat, log=True, batch_size=1024 * 32)
    b.save()

print(f"Loaded two indexes of size: {len(a):,}, {len(b):,}")

min_elements = min(len(a), len(b))
mean_similarity = 0.0
mean_recovered_similarity = 0.0

for i in range(min_elements):
    a_vec = a_mat[i]
    b_vec = b_mat[i]
    cos_similarity = dot(a_vec, b_vec) / (norm(a_vec) * norm(b_vec))
    mean_similarity += cos_similarity

    a_vec = a[i]
    b_vec = b[i]
    cos_similarity = dot(a_vec, b_vec) / (norm(a_vec) * norm(b_vec))
    mean_recovered_similarity += cos_similarity


mean_similarity /= min_elements
mean_recovered_similarity /= min_elements
print(
    f"Average vector similarity is {mean_similarity:.2f} in original dataset, and {mean_recovered_similarity:.2f} in recovered state in index"
)

secs, a_self_recall = measure_seconds(lambda: recall_members(a))
print(f"Self-recall of first index is {a_self_recall:.2f}, computed in {secs:.2f}s")

secs, b_self_recall = measure_seconds(lambda: recall_members(b))
print(f"Self-recall of first index is {b_self_recall:.2f}, computed in {secs:.2f}s")

secs, bimapping = measure_seconds(lambda: a.join(b, max_proposals=20))
intersection_size = len(bimapping)
recall = 0
for i, j in bimapping.items():
    recall += i == j

recall *= 100.0 / min_elements
print(f"Took {secs:.2f}s to find {intersection_size:,} intersections: {recall:.2f} %")
