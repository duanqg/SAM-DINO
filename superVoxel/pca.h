#include <iostream>
#include <fstream>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <map>

struct CustomPoint
{
    PCL_ADD_POINT4D;
    float scalar_Scalar_field;
    uint8_t r, g, b;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(CustomPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (uint8_t, r, r)
    (uint8_t, g, g)
    (uint8_t, b, b)
    (float, scalar_Scalar_field, scalar_Scalar_field)
)

void pca(const char* name){
    pcl::PointCloud<CustomPoint>::Ptr cloud(new pcl::PointCloud<CustomPoint>);
//    pcl::io::loadPLYFile("P1180160.ply", *cloud);

    std::ifstream in(name);
    int n_lines = 0;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream is(line);

        CustomPoint point;
        if (!(is >> point.x) || !(is >> point.y) || !(is >> point.z) || !(is >> point.scalar_Scalar_field)
             || !(is >> point.r) || !(is >> point.g) || !(is >> point.b)) {
            LOG(INFO) << "Invalid XYZ format at line: " << n_lines++;
            in.close();
//            return false;
        }
        cloud->push_back(point);
    }

    in.close();

    std::map<float, pcl::PointCloud<CustomPoint>::Ptr> clusters;

    for (int i = 0; i < cloud->size(); ++i)
    {
        float scalar = cloud->points[i].scalar_Scalar_field;
        if (clusters.find(scalar) == clusters.end())
        {
            clusters[scalar] = pcl::PointCloud<CustomPoint>::Ptr(new pcl::PointCloud<CustomPoint>);
        }
        clusters[scalar]->push_back(cloud->points[i]);
    }

    std::ofstream outputFile1("result_Plane.xyz");
    std::ofstream outputFile2("result_Curve.xyz");
    std::ofstream outputFile3("result_Broken.xyz");

    for (auto it = clusters.begin(); it != clusters.end(); ++it)
    {
        pcl::PointCloud<CustomPoint>::Ptr cluster = it->second;

        Eigen::Vector4f pcaCentroid;
        pcl::compute3DCentroid(*cluster, pcaCentroid);
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cluster, pcaCentroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
        Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();


        float lambda1 = eigenValuesPCA(0);
        float lambda2 = eigenValuesPCA(1);
        float lambda3 = eigenValuesPCA(2);
        float sum_of_eigenvalues = lambda1 + lambda2 + lambda3;

        float eigen_ratio = lambda1 / sum_of_eigenvalues;

        if (eigen_ratio < 0.005 && lambda2/lambda1>=10 && lambda3/lambda1>=10) {
            outputFile1 << "Plane " << pcaCentroid[0] << " " << pcaCentroid[1] << " " << pcaCentroid[2] << std::endl;
        }
        else if (abs(eigen_ratio-0.333) < 0.1) {
            outputFile3 << "Broken " << pcaCentroid[0] << " " << pcaCentroid[1] << " " << pcaCentroid[2] << std::endl;
        }
        else {
            outputFile2 << "Curve " << pcaCentroid[0] << " " << pcaCentroid[1] << " " << pcaCentroid[2] << std::endl;
        }

    }
    outputFile1.close();
    outputFile2.close();
    outputFile3.close();
}