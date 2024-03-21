#include "codelibrary/base/log.h"
#include "codelibrary/geometry/io/xyz_io.h"
#include "codelibrary/geometry/point_cloud/pca_estimate_normals.h"
#include "codelibrary/geometry/point_cloud/supervoxel_segmentation.h"
#include "codelibrary/geometry/util/distance_3d.h"
#include "codelibrary/util/tree/kd_tree.h"

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <iostream>
using namespace std;

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr points_pcl(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

/// Point with Normal.
struct PointWithNormal : cl::RPoint3D {
    PointWithNormal() {}

    cl::RVector3D normal;
};

/**
 * Metric used in VCCS supervoxel segmentation.
 *
 * Reference:
 *   Rusu, R.B., Cousins, S., 2011. 3d is here: Point cloud library (pcl),
 *   IEEE International Conference on Robotics and Automation, pp. 1–4.
 */
class VCCSMetric {
public:
    explicit VCCSMetric(double resolution)
            : resolution_(resolution) {}

    double operator() (const PointWithNormal& p1,
                       const PointWithNormal& p2) const {
        return 1.0 - std::fabs(p1.normal * p2.normal) +
               cl::geometry::Distance(p1, p2) / resolution_ * 0.4;
    }

private:
    double resolution_;
};

/**
 * Save point clouds (with segmentation colors) into the file.
 */
void WritePoints(const char* filename,
                 int n_supervoxels,
                 const cl::Array<cl::RPoint3D>& points,
                 const cl::Array<int>& labels) {
    cl::Array<cl::RGB32Color> colors(points.size());
    std::mt19937 random;
    cl::Array<cl::RGB32Color> supervoxel_colors(n_supervoxels);
    for (int i = 0; i < n_supervoxels; ++i) {
        supervoxel_colors[i] = cl::RGB32Color(random());
    }
    for (int i = 0; i < points.size(); ++i) {
        colors[i] = supervoxel_colors[labels[i]];
//            LOG(INFO) << "The points are written into " << labels[i];
    }

    if (cl::geometry::io::WriteXYZPoints(filename, points, colors, labels)) {
        LOG(INFO) << "The points are written into " << filename;
    }
    //    pcl::io::savePCDFile<pcl::PointXYZ> ("../save_path/xx.pcd", *cloud);
}

int main() {
    LOG_ON(INFO);
    const std::string filename = "../../data/result_pcd/P1180184";
    const string file_type_input =  ".ply";
    const string file_type_output = ".xyz";
    const string filename_str = filename;
    const string ply_filename = filename_str + file_type_input;
    if (pcl::io::loadPLYFile<pcl::PointXYZRGBNormal>(ply_filename, *points_pcl) == -1) //* load the file
    {
        LOG(INFO) << "Please check if " << ply_filename << " is exist.";
        return (-1);
    }else {
        //成功导入
        LOG(INFO) << "ply src loads ok!";
    }
    size_t n_points = points_pcl->size();
    cl::Array<PointWithNormal> oriented_points_cgal(n_points);
    cl::Array<cl::RPoint3D> points(n_points);
    LOG(INFO) << n_points << " points are imported.";
    for (int i = 0; i < n_points; ++i) {
        oriented_points_cgal[i].x = points_pcl->points[i].PointXYZRGBNormal::x;
        oriented_points_cgal[i].y = points_pcl->points[i].PointXYZRGBNormal::y;
        oriented_points_cgal[i].z = points_pcl->points[i].PointXYZRGBNormal::z;
        points[i].x = points_pcl->points[i].PointXYZRGBNormal::x;
        points[i].y = points_pcl->points[i].PointXYZRGBNormal::y;
        points[i].z = points_pcl->points[i].PointXYZRGBNormal::z;
        cl::RVector3D normal;
        normal[0] =  points_pcl->points[i].PointXYZRGBNormal::normal[0];
        normal[1] =  points_pcl->points[i].PointXYZRGBNormal::normal[1];
        normal[2] =  points_pcl->points[i].PointXYZRGBNormal::normal[2];
        oriented_points_cgal[i].normal = normal;
    }

    LOG(INFO) << "Finish import xyz, normal";
    LOG(INFO) << "Building KD tree...";
    cl::KDTree<cl::RPoint3D> kdtree;
    kdtree.SwapPoints(&points);

    const int k_neighbors = 20;
    assert(k_neighbors < n_points);

    LOG(INFO) << "Compute the k-nearest neighbors for each point";


    //    cl::Array<cl::RVector3D> normals(n_points);
    cl::Array<cl::Array<int>> neighbors(n_points);
    //    cl::Array<cl::RPoint3D> neighbor_points(k_neighbors);
    for (int i = 0; i < n_points; ++i) {
        kdtree.FindKNearestNeighbors(kdtree.points()[i], k_neighbors,
                                     &neighbors[i]);
        //        for (int k = 0; k < k_neighbors; ++k) {
        //            neighbor_points[k] = kdtree.points()[neighbors[i][k]];
        //        }
        //        cl::geometry::point_cloud::PCAEstimateNormal(neighbor_points.begin(),
        //                                                     neighbor_points.end(),
        //                                                     &normals[i]);
    }
    //    for (int i = 0; i < n_points; ++i) {
    //        oriented_points[i].x = points[i].x;
    //        oriented_points[i].y = points[i].y;
    //        oriented_points[i].z = points[i].z;
    //        oriented_points_cgal[i].normal = normals[i];
    //    }
    kdtree.SwapPoints(&points);

    LOG(INFO) << "Start supervoxel segmentation...";

    // NOTE!!! Change the resolution to get variable sized supervoxels.
    const double resolution = 0.1;

    VCCSMetric metric(resolution);
    cl::Array<int> labels, supervoxels;
    cl::geometry::point_cloud::SupervoxelSegmentation(oriented_points_cgal, neighbors, resolution, metric, &supervoxels, &labels);

    int n_supervoxels = supervoxels.size();
    LOG(INFO) << n_supervoxels << " supervoxels computed.";
    WritePoints((filename_str + file_type_output).c_str(), n_supervoxels, points, labels);
    return 0;
}
