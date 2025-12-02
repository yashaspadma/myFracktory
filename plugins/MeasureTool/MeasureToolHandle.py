# Copyright (c) 2023 Aldo Hoeben / fieldOfView
# MeasureTool is released under the terms of the AGPLv3 or higher.

from UM.Mesh.MeshData import MeshData, calculateNormalsFromIndexedVertices
from UM.Scene.ToolHandle import ToolHandle
from UM.Math.Vector import Vector
from UM.View.GL.OpenGL import OpenGL
from UM.Resources import Resources
from UM.Math.Color import Color

import trimesh
import numpy

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .MeasureTool import MeasureTool


class MeasureToolHandle(ToolHandle):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._name = "MeasureToolHandle"

        self._handle_width = 2
        self._selection_mesh = MeshData()
        self._line_mesh = None  # type: Optional[MeshData]

        self._tool = None  # type: Optional[MeasureTool]

    def setTool(self, tool: "MeasureTool") -> None:
        self._tool = tool

    def buildMesh(self) -> None:
        mesh = self._toMeshData(
            trimesh.creation.icosphere(subdivisions=2, radius=self._handle_width / 2)
        )
        self.setSolidMesh(mesh)
        
    def _buildLineMesh(self, point_a: Vector, point_b: Vector) -> MeshData:
        """Create a cylinder mesh to represent a line between two points."""
        # Calculate the direction and distance
        direction = point_b - point_a
        distance = direction.length()
        
        if distance < 0.01:  # If points are too close, don't draw a line
            return MeshData()
        
        # Create a cylinder
        cylinder = trimesh.creation.cylinder(
            radius=0.2,  # Line thickness
            height=distance,
            sections=8
        )
        
        # Calculate rotation to align cylinder with the direction vector
        # Default cylinder is along Z axis, we need to rotate it
        direction_normalized = direction.normalized()
        z_axis = numpy.array([0, 0, 1])
        direction_array = numpy.array([direction_normalized.x, direction_normalized.y, direction_normalized.z])
        
        # Calculate rotation axis and angle
        rotation_axis = numpy.cross(z_axis, direction_array)
        rotation_axis_length = numpy.linalg.norm(rotation_axis)
        
        if rotation_axis_length > 0.001:  # Avoid division by zero
            rotation_axis = rotation_axis / rotation_axis_length
            angle = numpy.arccos(numpy.clip(numpy.dot(z_axis, direction_array), -1.0, 1.0))
            
            # Create rotation matrix
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)
            cylinder.apply_transform(rotation_matrix)
        
        # Translate to midpoint
        midpoint = (point_a + point_b) * 0.5
        translation = numpy.array([midpoint.x, midpoint.y, midpoint.z])
        cylinder.apply_translation(translation)
        
        return self._toMeshData(cylinder)

    def render(self, renderer) -> bool:
        if not self._shader:
            # Use color shader instead of toolhandle shader to support custom colors
            self._shader = OpenGL.getInstance().createShaderProgram(
                Resources.getPath(Resources.Shaders, "color.shader")
            )
            # Set green color for the measurement tool
            self._shader.setUniformValue("u_color", Color(0.0, 1.0, 0.0, 1.0))

        if self._auto_scale:
            active_camera = self._scene.getActiveCamera()
            if active_camera.isPerspective():
                camera_position = active_camera.getWorldPosition()
                dist = (camera_position - self.getWorldPosition()).length()
                scale = dist / 400
            else:
                view_width = active_camera.getViewportWidth()
                current_size = view_width + (
                    2 * active_camera.getZoomFactor() * view_width
                )
                scale = current_size / view_width * 5

            self.setScale(Vector(scale, scale, scale))

        if self._solid_mesh and self._tool:
            # Render the two dots at point A and point B
            for position in [self._tool.getPointA(), self._tool.getPointB()]:
                self.setPosition(Vector(position.x(), position.y(), position.z()))
                renderer.queueNode(
                    self, mesh=self._solid_mesh, overlay=False, shader=self._shader
                )
            
            # Render the line between the two points
            point_a = self._tool.getPointA()
            point_b = self._tool.getPointB()
            point_a_vec = Vector(point_a.x(), point_a.y(), point_a.z())
            point_b_vec = Vector(point_b.x(), point_b.y(), point_b.z())
            
            # Only draw line if points are different
            if (point_a_vec - point_b_vec).length() > 0.01:
                self._line_mesh = self._buildLineMesh(point_a_vec, point_b_vec)
                if self._line_mesh and self._line_mesh.getVertexCount() > 0:
                    # Reset position for line (it's already positioned in world space)
                    self.setPosition(Vector(0, 0, 0))
                    self.setScale(Vector(1, 1, 1))
                    renderer.queueNode(
                        self, mesh=self._line_mesh, overlay=False, shader=self._shader
                    )

        return True

    def _toMeshData(self, tri_node: trimesh.base.Trimesh) -> MeshData:
        tri_faces = tri_node.faces
        tri_vertices = tri_node.vertices

        indices = []
        vertices = []

        index_count = 0
        face_count = 0
        for tri_face in tri_faces:
            face = []
            for tri_index in tri_face:
                vertices.append(tri_vertices[tri_index])
                face.append(index_count)
                index_count += 1
            indices.append(face)
            face_count += 1

        vertices = numpy.asarray(vertices, dtype=numpy.float32)
        indices = numpy.asarray(indices, dtype=numpy.int32)
        normals = calculateNormalsFromIndexedVertices(vertices, indices, face_count)

        mesh_data = MeshData(vertices=vertices, indices=indices, normals=normals)

        return mesh_data
