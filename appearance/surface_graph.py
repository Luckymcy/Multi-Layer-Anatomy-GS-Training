"""Fixed CT-surface geodesic graph and differentiable robust RGB regularizer."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def bounded_distances(indptr, indices, weights, start, targets, limit):
    """Sparse bounded Dijkstra, without allocating a mesh-size dense distance row."""
    remaining, result = set(targets), {}
    distance, queue = {int(start): 0.}, [(0., int(start))]
    while queue and remaining:
        d, node = heapq.heappop(queue)
        if d != distance[node]:
            continue
        if node in remaining:
            result[node] = d
            remaining.remove(node)
        for offset in range(indptr[node], indptr[node+1]):
            other, candidate = int(indices[offset]), d + float(weights[offset])
            if candidate <= limit and candidate < distance.get(other, np.inf):
                distance[other] = candidate
                heapq.heappush(queue, (candidate, other))
    return result


def build_graph(xyz, vertices, faces, normals, neighbors=8, radius=2.5,
                max_surface_distance=1.5, normal_angle=45.):
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    xyz, vertices, normals = (np.asarray(a, dtype=np.float64) for a in (xyz, vertices, normals))
    faces = np.asarray(faces, dtype=np.int64)
    if (neighbors < 1 or radius <= 0 or max_surface_distance <= 0 or not 0 < normal_angle < 90
            or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2
            or vertices.ndim != 2 or vertices.shape[1] != 3 or normals.shape != vertices.shape
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.min() < 0 or faces.max() >= len(vertices)
            or any(not np.isfinite(a).all() for a in (xyz, vertices, normals))):
        raise ValueError('Invalid surface/graph parameters')
    norm = np.linalg.norm(normals, axis=1)
    if (norm < 1e-8).any():
        raise ValueError('Surface normals must be nonzero')
    normals = normals / norm[:,None]
    distance, anchors = cKDTree(vertices).query(xyz)
    mapped = distance <= max_surface_distance
    # Mesh adjacency is derived only from actual triangle edges, never XYZ proximity.
    mesh_edges = np.unique(np.sort(np.concatenate([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]]),axis=1),axis=0)
    lengths = np.linalg.norm(vertices[mesh_edges[:,0]]-vertices[mesh_edges[:,1]],axis=1)
    adjacency = csr_matrix((np.tile(lengths,2),
        (np.concatenate([mesh_edges[:,0],mesh_edges[:,1]]),
         np.concatenate([mesh_edges[:,1],mesh_edges[:,0]]))),shape=(len(vertices),len(vertices)))
    components, membership = connected_components(adjacency, directed=False)
    d, nearby = cKDTree(xyz).query(xyz,k=min(neighbors+1,len(xyz)),distance_upper_bound=radius)
    source = np.repeat(np.arange(len(xyz)), d.shape[1])
    target, distances = nearby.ravel(), d.ravel()
    valid = np.isfinite(distances) & (target < len(xyz)) & (source != target)
    edges = np.unique(np.sort(np.column_stack([source[valid],target[valid]]),axis=1),axis=0)
    candidates = len(edges)
    edges = edges[mapped[edges].all(1)]
    a, b = anchors[edges[:,0]], anchors[edges[:,1]]
    cosine = (normals[a]*normals[b]).sum(1)
    valid = (membership[a] == membership[b]) & (cosine >= np.cos(np.deg2rad(normal_angle)))
    edges, a, b = edges[valid], a[valid], b[valid]
    groups = {}
    for index, node in enumerate(a):
        groups.setdefault(int(node), []).append(index)
    geodesic = np.full(len(edges),np.inf)
    for count, (node, rows) in enumerate(groups.items()):
        targets = b[rows]
        distances = bounded_distances(adjacency.indptr, adjacency.indices, adjacency.data,
                                      node, targets, radius)
        geodesic[rows] = [distances.get(int(t),np.inf) for t in targets]
        if count and count % 3000 == 0:
            print(f'Checked {count}/{len(groups)} surface anchors',flush=True)
    keep = np.isfinite(geodesic)
    edges, geodesic = edges[keep], geodesic[keep]
    if not len(edges):
        raise ValueError('No reliable surface edges; check coordinate system and surface reference')
    weight = np.exp(-.5*(geodesic/radius)**2)
    degree = np.bincount(edges.ravel(),minlength=len(xyz))
    summary = dict(gaussians=len(xyz), mapped_gaussians=int(mapped.sum()),
                   connected_gaussians=int((degree>0).sum()), isolated_gaussians=int((degree==0).sum()),
                   candidate_edges=candidates, accepted_edges=len(edges), mesh_components=components,
                   surface_distance_p50_p95_max=np.percentile(distance,[50,95,100]).tolist(),
                   accepted_geodesic_max=float(geodesic.max()), radius_mm=radius,
                   max_surface_distance_mm=max_surface_distance, normal_angle_degrees=normal_angle,
                   neighbors=neighbors, coordinate_system='LPS_mm',
                   limitation='CT surface prior; nearest-vertex projection is approximate. Unmapped/isolated Gaussians receive no graph gradient.')
    return edges, weight.astype(np.float32), summary


class SurfaceColorGraph(nn.Module):
    def __init__(self, edges, weights, count, delta=.05):
        super().__init__()
        edges, weights = torch.as_tensor(edges,dtype=torch.long), torch.as_tensor(weights,dtype=torch.float32)
        if (count < 2 or delta <= 0 or edges.ndim != 2 or edges.shape[1] != 2 or not len(edges)
                or edges.min() < 0 or edges.max() >= count or (edges[:,0] == edges[:,1]).any()
                or weights.shape != (len(edges),) or not torch.isfinite(weights).all() or (weights<=0).any()):
            raise ValueError('Invalid regularization graph')
        self.register_buffer('edges',edges)
        self.register_buffer('weights',weights)
        self.count, self.delta = count, delta

    def forward(self, rgb):
        if rgb.shape != (self.count,3):
            raise ValueError('Graph/RGB row mismatch')
        difference = (rgb[self.edges[:,0]]-rgb[self.edges[:,1]]).abs()
        robust = torch.where(difference < self.delta, .5*difference.square()/self.delta,
                             difference-.5*self.delta).mean(-1)
        return (self.weights*robust).sum()/self.weights.sum()

    @classmethod
    def load(cls, path, model_hash, row_indices, delta=.05):
        with np.load(path,allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['model_sha256'] != model_hash:
                raise ValueError('Surface graph/source model fingerprint mismatch')
            np.testing.assert_array_equal(data['ply_row_indices'],row_indices.detach().cpu().numpy())
            graph = cls(data['edges'],data['weights'],len(row_indices),delta)
        return graph, metadata


def main():
    from plyfile import PlyData
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ply',type=Path,required=True)
    ap.add_argument('--surface',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--layer',type=int,default=4)
    ap.add_argument('--radius-mm',type=float,default=2.5)
    ap.add_argument('--max-surface-distance-mm',type=float,default=1.5)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Use a new graph path')
    vertex = PlyData.read(str(args.ply))['vertex'].data
    rows = np.flatnonzero(vertex['layer']==args.layer)
    xyz = np.column_stack([vertex[k][rows] for k in ('x','y','z')])
    with np.load(args.surface,allow_pickle=False) as surface:
        surface_metadata = json.loads(str(surface['metadata']))
        if surface_metadata.get('coordinate_system') != 'LPS_mm':
            raise ValueError('Expected an explicitly physical LPS/mm surface')
        edges, weights, metadata = build_graph(xyz,surface['vertices'],surface['faces'],surface['normals'],
            radius=args.radius_mm,max_surface_distance=args.max_surface_distance_mm)
        metadata['surface_metadata'] = surface_metadata
    metadata.update(model_sha256=digest(args.ply), surface_sha256=digest(args.surface),layer=args.layer)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,edges=edges,weights=weights,ply_row_indices=rows,
                        metadata=json.dumps(metadata))
    print(json.dumps(metadata,indent=2),flush=True)


if __name__ == '__main__':
    main()
