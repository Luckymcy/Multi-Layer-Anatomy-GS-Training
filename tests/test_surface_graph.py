import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from appearance.surface_graph import build_graph, SurfaceColorGraph


def folded_surface():
    # Two close sheets connected only at the far end. Euclidean KNN would bridge them.
    vertices = np.array([[x,y,z] for z in (0.,.1) for x in range(5) for y in range(2)])
    faces = []
    for offset in (0,10):
        for x in range(4):
            a = offset+2*x
            faces.extend([[a,a+2,a+1],[a+1,a+2,a+3]])
    faces.extend([[8,18,9],[9,18,19]])
    normals = np.tile([0.,0.,1.],(len(vertices),1))
    return vertices,np.array(faces),normals


class SurfaceGraphTests(unittest.TestCase):
    def test_geodesic_rejects_nearby_folded_surface(self):
        vertices,faces,normals = folded_surface()
        xyz = vertices[[0,2,10,12]]
        edges,weights,report = build_graph(xyz,vertices,faces,normals,radius=1.5)
        self.assertEqual(set(map(tuple,edges)),{(0,1),(2,3)})
        self.assertEqual(report['mesh_components'],1)
        self.assertTrue((weights>0).all())

    def test_unmapped_point_remains_isolated(self):
        vertices,faces,normals = folded_surface()
        xyz = np.array([[0.,0.,0.],[1.,0.,0.],[100.,100.,100.]])
        edges,_,report = build_graph(xyz,vertices,faces,normals,radius=1.5)
        self.assertEqual(edges.tolist(),[[0,1]])
        self.assertEqual(report['mapped_gaussians'],2)
        self.assertEqual(report['isolated_gaussians'],1)

    def test_opposite_normals_are_not_connected(self):
        vertices,faces,normals = folded_surface()
        normals[2] = [0.,0.,-1.]
        with self.assertRaises(ValueError):
            build_graph(vertices[[0,2]],vertices,faces,normals,radius=1.5)

    def test_color_gradient_reduces_loss_and_respects_components(self):
        graph = SurfaceColorGraph([[0,1],[2,3]],[1.,1.],5)
        colors = torch.tensor([[.2]*3,[.6]*3,[.8]*3,[.8]*3,[.1]*3],requires_grad=True)
        before = graph(colors)
        before.backward()
        self.assertGreater(float(colors.grad[:2].norm()),0)
        self.assertEqual(float(colors.grad[2:].abs().sum()),0)
        after = graph(colors.detach()-.1*colors.grad)
        self.assertLess(float(after),float(before))
        self.assertFalse(any(b.requires_grad for b in graph.buffers()))

    def test_constant_color_zero_and_density_normalization(self):
        graph = SurfaceColorGraph([[0,1]],[1.],2)
        self.assertEqual(float(graph(torch.full((2,3),.4))),0.)
        repeated = SurfaceColorGraph([[0,1],[0,1]],[1.,1.],2)
        colors = torch.tensor([[.1]*3,[.8]*3])
        torch.testing.assert_close(graph(colors),repeated(colors))

    def test_saved_graph_binds_model_and_row_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'graph.npz'
            np.savez(path,edges=np.array([[0,1]]),weights=np.array([1.]),
                     ply_row_indices=np.array([4,9]),metadata=json.dumps({'model_sha256':'model'}))
            graph,_=SurfaceColorGraph.load(path,'model',torch.tensor([4,9]))
            self.assertEqual(graph.count,2)
            with self.assertRaises(ValueError):
                SurfaceColorGraph.load(path,'different',torch.tensor([4,9]))
            with self.assertRaises(AssertionError):
                SurfaceColorGraph.load(path,'model',torch.tensor([9,4]))


if __name__ == '__main__':
    unittest.main()
