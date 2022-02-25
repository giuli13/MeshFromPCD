"""
=============================
Written by: Giulia Martinelli
=============================
==================
About this Script:
==================

This script file defines three possible plot:
 - Mesh extracted from point cloud
 - Skeleton extracted from the mesh (format OPENPOSE BODY25)
 - Point Cloud filterd for the mesh extraction

All can be plotted together

"""

import os
import numpy as np
import open3d as o3d
import natsort
import argparse
import time
import threading
import joblib



smpl_mapping = [24,12,17,19,21,16,18,20,0,2,5,8,1,4,7,25,26,27,28,29,30,31,32,33,34]
smpl_connections = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,24],
                    [11,22],[22,23],[8,12],[12,13],[13,14],[14,21],[14,19],[19,20],[0,15],[15,17],[0,16],[16,18]]

colors = [[1, 0, 0] for i in range(len(smpl_connections))]



def main():
    
    # Read data: insert your path to the mesh, pcd and to the pickle file containing smpl parameters
    mesh = o3d.io.read_triangle_mesh('/home/giuliamartinelli/Documents/Code/unsupervised3dhuman/demo/kinect_ply_output/Frame001_EM.obj')
    pcd = o3d.io.read_point_cloud('/home/giuliamartinelli/Documents/Code/unsupervised3dhuman/demo/kinect_ply_output/Frame001.ply')
    pickle_file = r"/home/giuliamartinelli/Documents/Code/unsupervised3dhuman/demo/kinect_ply_output/Frame001_EM.pkl"
    joints = joblib.load(pickle_file)['joints3d'][smpl_mapping]
    
    mesh.compute_vertex_normals()

    vert = mesh.vertices
    points = pcd.points
    center_pcd = pcd.get_center()
    center = mesh.get_center()
    mesh.vertices = o3d.utility.Vector3dVector(vert - center)
    pcd.points = o3d.utility.Vector3dVector(points - center_pcd)
    keypoints = o3d.geometry.PointCloud()
    keypoints.points = o3d.utility.Vector3dVector(joints)
    keypoints_center = keypoints.get_center()
    keypoints.points = o3d.utility.Vector3dVector(joints - keypoints_center)
    skeleton_joints = o3d.geometry.LineSet()
    skeleton_joints.points = o3d.utility.Vector3dVector(joints)
    center_skel = skeleton_joints.get_center()
    skeleton_joints.points = o3d.utility.Vector3dVector(joints-center_skel)
    skeleton_joints.lines = o3d.utility.Vector2iVector(smpl_connections)
    skeleton_joints.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    
    vis.create_window()
    # Add mesh, pcd, skeleton/keypoints to the visualizer
    # Comment for single visualization
    vis.add_geometry(mesh)
    vis.add_geometry(pcd)

    # This plot the entire skeleton
    vis.add_geometry(skeleton_joints)
    vis.add_geometry(keypoints)
    
    # control = vis.get_view_control()
    # control.set_zoom(1.5)
    for i in range(1, 76):
        # Read frame: insert your path to the mesh, pcd and to the pickle file containing smpl parameters
        filename = f'/home/giuliamartinelli/Documents/Code/unsupervised3dhuman/demo/kinect_ply_output/Frame{i:03d}_EM.obj'
        filepcd = f'/home/giuliamartinelli/Documents/Code/unsupervised3dhuman/demo/kinect_ply/Frame{i:03d}.ply'
        filejoints = f'/home/giuliamartinelli/Documents/Code/unsupervised3dhuman/demo/kinect_ply_output/Frame{i:03d}_EM.pkl'

        new_mesh = o3d.io.read_triangle_mesh(filename)
        new_pcd = o3d.io.read_point_cloud(filepcd)
        new_joints = joblib.load(filejoints)['joints3d'][smpl_mapping]
        print(filename)
        vert = new_mesh.vertices
        center = new_mesh.get_center()
        center_pcd = new_pcd.get_center()
        mesh.vertices = o3d.utility.Vector3dVector(vert - center)
        pcd.points = o3d.utility.Vector3dVector(new_pcd.points - center)
        skeleton_joints.points = o3d.utility.Vector3dVector(new_joints - center)
        keypoints.points = o3d.utility.Vector3dVector(new_joints - center)

        # Add mesh, pcd, skeleton/keypoints to the visualizer
        # Comment for single visualization
        vis.update_geometry(mesh)
        vis.update_geometry(pcd)
        # This plot the entire skeleton
        vis.update_geometry(skeleton_joints)
        vis.update_geometry(keypoints)
        
        vis.update_renderer()
        vis.poll_events()

        time.sleep(0.5)
    
    vis.run()

    
if __name__ == '__main__':
    main()