#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/parula.h>
#include <igl/per_corner_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include "io_tools.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/geometry/TriangleMesh.h"

int main(int argc, char *argv[])
{
  using namespace Eigen;
  std::string filename = "../data/00/velodyne/000000.bin";
  if(argc>1)
  {
    filename = argv[1];
  }
  else
  {
    std::printf("Expected test pointcloud file, for example: %s\n",filename.c_str());
    exit(0);
  }
  std::vector<Eigen::Vector3d> pointcloud_arr;
  readPointCloud(filename, pointcloud_arr);
  auto pointcloud = open3d::geometry::PointCloud(std::move(pointcloud_arr));
  pointcloud.EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.6, 30));
  std::shared_ptr<open3d::geometry::TriangleMesh> mesh;
  std::tie(mesh, std::ignore) = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(pointcloud);
  Eigen::MatrixXd Vtr = Eigen::Map<Eigen::MatrixXd>(reinterpret_cast<double*>(mesh->vertices_.data()), 3, mesh->vertices_.size());  // Vertices
  Eigen::MatrixXi Ftr = Eigen::Map<Eigen::MatrixXi>(reinterpret_cast<int*>(mesh->triangles_.data()), 3, mesh->triangles_.size());  // Faces (Connectity of Vertices as Triangles)
  auto const &V = Vtr.transpose();  // N, 3
  auto const &F = Ftr.transpose();  // N, 3
  std::printf("V:(%ld,%ld), F:(%ld,%ld)\n", V.rows(), V.cols(), F.rows(), F.cols());
  // Alternative discrete mean curvature
  MatrixXd HN;
  SparseMatrix<double> L,M,Minv;
  igl::cotmatrix(V,F,L);
  igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_VORONOI,M);
  igl::invert_diag(M,Minv);
  // Laplace-Beltrami of position
  HN = -Minv*(L*V);
  // Extract magnitude as mean curvature
  VectorXd H = HN.rowwise().norm();

  // Compute curvature directions via quadric fitting
  MatrixXd PD1,PD2;
  VectorXd PV1,PV2;
  igl::principal_curvature(V,F,PD1,PD2,PV1,PV2);
  // mean curvature
  H = 0.5*(PV1+PV2);

  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);

  viewer.data().set_data(H);

  // Average edge length for sizing
  const double avg = igl::avg_edge_length(V,F);

  // Draw a red segment parallel to the maximal curvature direction
  const RowVector3d red(0.8,0.2,0.2),blue(0.2,0.2,0.8);
  viewer.data().add_edges(V + PD1*avg, V - PD1*avg, red);

  // Draw a blue segment parallel to the minimal curvature direction
  viewer.data().add_edges(V + PD2*avg, V - PD2*avg, blue);

  // Hide wireframe
  viewer.data().show_lines = false;

  viewer.launch();
}