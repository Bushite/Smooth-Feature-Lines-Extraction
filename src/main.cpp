#include <igl/readOFF.h>
#include <igl/readOBJ.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/doublearea.h>
#include <igl/grad.h>
#include <igl/barycenter.h>
#include <igl/adjacency_list.h>
#include <igl/jet.h>
#include <igl/principal_curvature.h>
#include <Eigen/Core>
// STL libraries
#include <algorithm>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stack>

using namespace std;

#define epsilon 10e-6

// Vertex array, #V x3
Eigen::MatrixXd V;
// Face array, #F x3
Eigen::MatrixXi F;
// Vertex normals, #V x3
Eigen::MatrixXd N_vertices;
// Face normals, #F x3
Eigen::MatrixXd N_faces;
// Triangle-triangle adjacency
Eigen::MatrixXi TT;
Eigen::MatrixXi TTi;
// Vertex-triangle adjacency
vector<vector<int> > VT;
vector<vector<int> > VTi;
// Area for each triangle
Eigen::VectorXd A;
// Bary center for each triangle
Eigen::MatrixXd B;
// Adjacency list for each vertex
vector<vector<int> > VV;
// Gradient operator
Eigen::SparseMatrix<double> G;

// Shape operator fot vertices
vector<Eigen::Matrix3d> operator_vertices;

// Principal curvatures
Eigen::VectorXd k_max;
Eigen::VectorXd k_min;

// Unit length eigenvectors corresponding to k_max and k_min
Eigen::MatrixXd k_vec_max;
Eigen::MatrixXd k_vec_min;

// Extremality for each vertex
Eigen::VectorXd e_max;
Eigen::VectorXd e_min;

// Unordered set of regular triangles ans singular triangles
unordered_set<int> regular_max;
unordered_set<int> singular_max;
unordered_set<int> regular_min;
unordered_set<int> singular_min;

vector<Eigen::RowVector3i> regular_sign_max;
vector<Eigen::RowVector3i> regular_sign_min;

// Salient edges
vector<Eigen::MatrixXd> edges_max;
vector<Eigen::MatrixXd> edges_min;

igl::opengl::glfw::Viewer viewer;

// Compute the regularity of each triangle
void compute_triangle_regularity(
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& k_vec,
    unordered_set<int>& regular,
    unordered_set<int>& singular
    )
{
    int s[] = {1, -1};
    for (int f = 0; f < F.rows(); f++) {
        bool is_regular = false;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    Eigen::RowVector3d k_vec_p1 = s[i] * k_vec_max.row(F(f, 0));
                    Eigen::RowVector3d k_vec_p2 = s[j] * k_vec_max.row(F(f, 1));
                    Eigen::RowVector3d k_vec_p3 = s[k] * k_vec_max.row(F(f, 2));

                    if (k_vec_p1.dot(k_vec_p2) > 0 && k_vec_p2.dot(k_vec_p3) > 0 && k_vec_p3.dot(k_vec_p1) > 0)
                        is_regular = true;
                }
            }
        }

        if (is_regular)
            regular.insert(f);
        else
            singular.insert(f);
    }
}


// Compute the extremality of vertex p
double vertex_extremality(int p, Eigen::VectorXd& k, Eigen::MatrixXd& k_vec, Eigen::MatrixXd& G_k, unordered_set<int>& regular) {
    vector<int> star = VT[p];

    // Compute extremality of the vertex
    double e = 0;
    double area_sum = 0;
    for (int t = 0; t < star.size(); t++) {
        if (regular.find(star[t]) == regular.end()) continue;

        area_sum += A(star[t]);
        e += A(star[t])*G_k.row(star[t]).dot(k_vec.row(p));
    }

    if (area_sum != 0)
        e = e / area_sum;
    else
        e = 0;

    return e;
}


// Compute the contan weight between vertex p and q (vertex p is the center of the star)
float cotan_weight(int p, int q) {
    vector<int> star = VT[p];
    vector<int> faces;
    // Find the two faces that p and q are in first
    for (int t = 0; t < star.size(); t++) {
        for (int i = 0; i < 3; i++) {
            if (F(star[t], i) == q) {
                faces.push_back(star[t]);
            }
        }
    }

    // If the edge pq is a boundary edge, then there is no cotan weight
    if (faces.size() != 2) return 0;
    // Find the other two vertices on the two faces other than p or q
    int f1 = faces[0];
    int f2 = faces[1];
    int p1;
    int p2;
    for (int i = 0; i < 3; i++) {
        if (F(f1, i) != p && F(f1, i) != q)
            p1 = F(f1, i);
        if (F(f2, i) != p && F(f2, i) != q)
            p2 = F(f2, i);
    }

    // Compute cotan weight
    Eigen::Vector3d v1 = V.row(p).transpose();
    Eigen::Vector3d v2 = V.row(q).transpose();
    Eigen::Vector3d v3 = V.row(p1).transpose();
    Eigen::Vector3d v4 = V.row(p2).transpose();

    double cotan1 = ((v1-v3).dot((v2-v3))) / ((v1-v3).cross((v2-v3))).norm();
    double cotan2 = ((v1-v4).dot((v2-v4))) / ((v1-v4).cross((v2-v4))).norm();

    return (cotan1 + cotan2) / 2;
}

// Function that computes the shape operator for each vertex
void compute_extremality
    (
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXi& TT,
    Eigen::MatrixXd& N_faces,
    Eigen::MatrixXd& N_vertices,
    Eigen::VectorXd& k_max,
    Eigen::VectorXd& k_min,
    Eigen::MatrixXd& k_vec_max,
    Eigen::MatrixXd& k_vec_min,
    Eigen::VectorXd& e_max,    // Extremality corresponding to e_max
    Eigen::VectorXd& e_min    // Extremality corresponding to e_min
    )
{
    Eigen::MatrixXd N_edges(F.rows()*3, 3);
    vector<Eigen::Matrix3d> operator_edges;     // Shape operator for edges
    vector<Eigen::Matrix3d> operator_vertices;  // Shape operator for vertices
    unordered_map<int, vector<int> > vtoe;  // A map mapping a vertex to all the incident edges

    int counter = 0;

    for (int f = 0; f < F.rows(); f++) {
        for (int ei = 0; ei < F.cols(); ei++) {
            // Look up the opposite face
            int g = TT(f,ei);
            // Boundary edge
            if (g == -1) continue;

            // If the vertex of the edge has not been processed
            if (vtoe.find(F(f, ei)) == vtoe.end()) {
                vector<int> v;
                v.push_back(counter);
                vtoe[F(f, ei)] = v;
            }
            else {
                vtoe[F(f, ei)].push_back(counter);
            }

            /*** Start computing the shape operator for the edge ***/
            Eigen::RowVector3d eg = V.row(F(f, (ei+1)%3)) - V.row(F(f, ei));
            double cos_value = N_faces.row(f).dot(N_faces.row(g)) / (N_faces.row(f).norm() * N_faces.row(g).norm());
            if (cos_value < -1) cos_value = -1;
            else if (cos_value > 1) cos_value = 1;
            double theta_e = M_PI - acos(cos_value);
            double He = 2 * eg.norm() * cos(theta_e / 2);

            Eigen::RowVector3d Ne = (N_faces.row(f) + N_faces.row(g)) / (N_faces.row(f) + N_faces.row(g)).norm();
            Eigen::Matrix3d Se = He * eg.normalized().cross(Ne).transpose() * eg.normalized().cross(Ne);

            N_edges.row(counter) = Ne;
            operator_edges.push_back(Se);

            counter++;
        }
    }

    // The size of N_edges should be #E x3
    N_edges.conservativeResize(counter, 3);

    /*** Start computing the shape operator for vertices ***/
    for (int v = 0; v < V.rows(); v++) {
        Eigen::Matrix3d sum = Eigen::MatrixXd::Constant(3, 3, 0);
        vector<int> edges = vtoe[v];
        for (int e = 0; e < edges.size(); e++) {
            sum += N_vertices.row(v).dot(N_edges.row(edges[e])) * operator_edges[edges[e]];
        }
        operator_vertices.push_back(sum / 2.0);
    }

    /*** Start computing the extremality for each vertex ***/
    k_max.resize(V.rows());
    k_min.resize(V.rows());

    k_vec_max.resize(V.rows(), 3);
    k_vec_min.resize(V.rows(), 3);

    // Compute eigen vectors and eigen values of the shape operator for each vertex
    for (int p = 0; p < V.rows(); p++) {
        Eigen::Matrix3d Sp = operator_vertices[p];
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sp);
        // Eigen values of Sp
        Eigen::VectorXd eVals = es.eigenvalues();
        // Eigen vectors of Sp
        Eigen::MatrixXd eVecs = es.eigenvectors();

        double minVal = min(abs(eVals(0)), min(abs(eVals(1)), abs(eVals(2))));
        int max_id, min_id;

        for (int i = 0; i < 3; i++) {
            if (abs(eVals(i)) == minVal) {
                if (eVals((i+1)%3) >= eVals((i+2)%3)) {
                    max_id = (i+1)%3;
                    min_id = (i+2)%3;
                }
                else {
                    max_id = (i+2)%3;
                    min_id = (i+1)%3;
                }

                break;
            }
        }

        k_max(p) = eVals(max_id);
        k_vec_max.row(p) = eVecs.row(max_id).normalized();
        k_min(p) = eVals(min_id);
        k_vec_min.row(p) = eVecs.row(min_id).normalized();
    }

    // Rescale k_max and k_min
    for (int p = 0; p < V.rows(); p++) {
        vector<int> star = VT[p];

        double area_sum = 0;
        for (int t = 0; t < star.size(); t++) {
            area_sum += abs(A(star[t]));
        }

        k_max(p) = 3.0/area_sum * k_max(p);
        k_min(p) = 3.0/area_sum * k_min(p);
    }

    // Or use libigl to compute the principal function
    // Note that the k_max and k_min are flipped here
    igl::principal_curvature(V, F, k_vec_min, k_vec_max, k_min, k_max);

    // Compute triangle regularity
    compute_triangle_regularity(F, k_vec_max, regular_max, singular_max);
    compute_triangle_regularity(F, k_vec_min, regular_min, singular_min);

    // Compute gradient of k_max and k_min
    Eigen::MatrixXd G_k_max = Eigen::Map<const Eigen::MatrixXd>((G*k_max).eval().data(),F.rows(),3);
    Eigen::MatrixXd G_k_min = Eigen::Map<const Eigen::MatrixXd>((G*k_min).eval().data(),F.rows(),3);

    e_max.resize(V.rows());
    e_min.resize(V.rows());
    for (int p = 0; p < V.rows(); p++) {
        e_max(p) = vertex_extremality(p, k_max, k_vec_max, G_k_max, regular_max);
        e_min(p) = vertex_extremality(p, k_min, k_vec_min, G_k_min, regular_min);
    }
}

// Detect the feature lines in the triangle
void feature_lines
    (
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::VectorXd& k_max,
    Eigen::VectorXd& k_min,
    Eigen::MatrixXd& k_vec_max,
    Eigen::MatrixXd& k_vec_min,
    Eigen::VectorXd& e_max,    // Extremality corresponding to e_max
    Eigen::VectorXd& e_min    // Extremality corresponding to e_min
    )
{
    edges_max = vector<Eigen::MatrixXd>();
    edges_min = vector<Eigen::MatrixXd>();

    vector<bool> faces_marked_min(F.rows(), false);
    vector<bool> faces_marked_max(F.rows(), false);

    /*** Extract feature lines on regular triangles ***/
    // For e_max
    for (int f = 0; f < F.rows(); f++) {
        // If the triangle is singular, skip it
        if (regular_max.find(f) == regular_max.end()) continue;

        // Start checking if the two conditions are saisfied
        Eigen::Matrix3d k_vec_max_T;
        Eigen::Vector3d e_max_T;

        Eigen::RowVector3i F_T(0, 1, 2);
        Eigen::MatrixXd V_T(3, 3);

        for (int i = 0; i < 3; i++) {
            // Build a copy of F and V containing only vertices on the triangle
            V_T.row(i) = V.row(F(f, i));

            // Choose k_vec and e consistently in T
            if (k_vec_max.row(F(f, i)).dot(k_vec_max.row(F(f, 0))) < 0) {
                k_vec_max_T.row(i) = -k_vec_max.row(F(f, i));
                e_max_T(i) = -e_max(F(f, i));
            }
            else {
                k_vec_max_T.row(i) = k_vec_max.row(F(f, i));
                e_max_T(i) = e_max(F(f, i));
            }
        }

        // Compute grad of e on the triangle
        Eigen::SparseMatrix<double> G_T;
        igl::grad(V_T, F_T, G_T);
        Eigen::RowVector3d G_e_max_T = Eigen::Map<const Eigen::MatrixXd>((G_T*e_max_T).eval().data(),1,3);

        // Compute the conditions needed first
        Eigen::RowVector3d k_vec_max_sum(0, 0, 0);
        double k_max_sum = 0;
        double k_min_sum = 0;

        for (int pi = 0; pi < 3; pi++) {
            k_vec_max_sum += k_vec_max_T.row(pi);
            k_max_sum += k_max(F(f, pi));
            k_min_sum += k_min(F(f, pi));
        }

        // If the two conditions are not satisfied
        if (!(G_e_max_T.dot(k_vec_max_sum) < 0 && abs(k_max_sum) > abs(k_min_sum)))
            continue;

        // Determine if any point on one edge is zero set
        vector<Eigen::RowVector3d> zero_set;
        for (int ei = 0; ei < F.cols(); ei++) {
            double e1_max = abs(e_max(F(f, ei))) < epsilon ? 0 : e_max(F(f, ei));
            if (k_vec_max.row(F(f, ei)).dot(k_vec_max.row(F(f, 0))) < 0) e1_max *= -1;
            double e2_max = abs(e_max(F(f, (ei+1)%3))) < epsilon ? 0 : e_max(F(f, (ei+1)%3));
            if (k_vec_max.row(F(f, (ei+1)%3)).dot(k_vec_max.row(F(f, 0))) < 0) e2_max *= -1;
            double e3_max = abs(e_max(F(f, (ei+2)%3))) < epsilon ? 0 : e_max(F(f, (ei+2)%3));
            if (k_vec_max.row(F(f, (ei+2)%3)).dot(k_vec_max.row(F(f, 0))) < 0) e3_max *= -1;

            // If all the e_max are identically zero
            if (e1_max == 0 && e2_max == 0 && e3_max == 0)
                continue;

            // If there is no zero level set on the edge
            if ((e1_max > 0 && e2_max > 0) || (e1_max < 0 && e2_max < 0))
                continue;

            // The zero set range [p1, p2), excluding p2
            Eigen::RowVector3d p1 = V.row(F(f, ei));
            Eigen::RowVector3d p2 = V.row(F(f, (ei+1)%3));

            if (e1_max == 0) {
                zero_set.push_back(p1);
                faces_marked_max[f] = true;
            }
            else {
                Eigen::RowVector3d p0 = (e1_max * p2 - e2_max * p1) / (e1_max - e2_max);
                // If the obtained point is not the other end point of the edge
                if (e2_max != 0) {
                    zero_set.push_back(p0);
                    faces_marked_max[f] = true;
                }
            }
        }

        if (zero_set.size() == 2) {
            Eigen::MatrixXd edge(2, 3);
            edge.row(0) = zero_set[0];
            edge.row(1) = zero_set[1];
            edges_max.push_back(edge);
        }
    }

    // For e_min
    for (int f = 0; f < F.rows(); f++) {
        // If the triangle is singular, skip it
        if (regular_min.find(f) == regular_min.end()) continue;

        // Start checking if the two conditions are saisfied
        Eigen::Matrix3d k_vec_min_T;
        Eigen::Vector3d e_min_T;

        Eigen::RowVector3i F_T(0, 1, 2);
        Eigen::MatrixXd V_T(3, 3);

        for (int i = 0; i < 3; i++) {
            // Build a copy of F and V containing only vertices on the triangle
            V_T.row(i) = V.row(F(f, i));

            // Choose k_vec and e consistently in T
            if (k_vec_min.row(F(f, i)).dot(k_vec_min.row(F(f, 0))) < 0) {
                k_vec_min_T.row(i) = -k_vec_min.row(F(f, i));
                e_min_T(i) = -e_min(F(f, i));
            }
            else {
                k_vec_min_T.row(i) = k_vec_min.row(F(f, i));
                e_min_T(i) = e_min(F(f, i));
            }
        }

        // Compute grad of e on the triangle
        Eigen::SparseMatrix<double> G_T;
        igl::grad(V_T, F_T, G_T);
        Eigen::RowVector3d G_e_min_T = Eigen::Map<const Eigen::MatrixXd>((G_T*e_min_T).eval().data(),1,3);

        // Compute the conditions needed first
        Eigen::RowVector3d k_vec_min_sum = Eigen::MatrixXd::Constant(1, 3, 0);
        double k_max_sum = 0;
        double k_min_sum = 0;

        for (int pi = 0; pi < 3; pi++) {
            k_vec_min_sum += k_vec_min_T.row(pi);
            k_max_sum += k_max(F(f, pi));
            k_min_sum += k_min(F(f, pi));
        }

        // If the two conditions are not satisfied
        if (!(G_e_min_T.dot(k_vec_min_sum) > 0 && abs(k_max_sum) < abs(k_min_sum)))
            continue;

        // Determine if any point on one edge is zero set
        vector<Eigen::RowVector3d> zero_set;
        for (int ei = 0; ei < F.cols(); ei++) {
            double e1_min = abs(e_min(F(f, ei))) < epsilon ? 0 : e_min(F(f, ei));
            if (k_vec_min.row(F(f, ei)).dot(k_vec_min.row(F(f, 0))) < 0) e1_min *= -1;
            double e2_min = abs(e_min(F(f, (ei+1)%3))) < epsilon ? 0 : e_min(F(f, (ei+1)%3));
            if (k_vec_min.row(F(f, (ei+1)%3)).dot(k_vec_min.row(F(f, 0))) < 0) e2_min *= -1;
            double e3_min = abs(e_min(F(f, (ei+2)%3))) < epsilon ? 0 : e_min(F(f, (ei+2)%3));
            if (k_vec_min.row(F(f, (ei+2)%3)).dot(k_vec_min.row(F(f, 0))) < 0) e3_min *= -1;

            // If all the e_min are identically zero
            if (e1_min == 0 && e2_min == 0 && e3_min == 0)
                continue;

            // If there is no zero level set on the edge
            if ((e1_min > 0 && e2_min > 0) || (e1_min < 0 && e2_min < 0))
                continue;

            // The zero set range [p1, p2), excluding p2
            Eigen::RowVector3d p1 = V.row(F(f, ei));
            Eigen::RowVector3d p2 = V.row(F(f, (ei+1)%3));

            if (e1_min == 0) {
                zero_set.push_back(p1);
                faces_marked_min[f] = true;
            }
            else {
                Eigen::RowVector3d p0 = (e1_min * p2 - e2_min * p1) / (e1_min - e2_min);
                // If the obtained point is not the other end point of the edge
                if (e2_min != 0) {
                    zero_set.push_back(p0);
                    faces_marked_min[f] = true;
                }
            }
        }

        if (zero_set.size() == 2) {
            Eigen::MatrixXd edge(2, 3);
            edge.row(0) = zero_set[0];
            edge.row(1) = zero_set[1];
            edges_min.push_back(edge);
        }
    }

    /*** Extract feature lines on singular triangles ***/
    // For e_max
    for (int f = 0; f < F.rows(); f++) {
        // If the triangle is regular, skip it
        if (regular_max.find(f) != regular_max.end()) continue;

        // Determine if any point on one edge is zero set
        vector<Eigen::RowVector3d> zero_set;
        for (int ei = 0; ei < F.cols(); ei++) {
            // Look up the opposite face
            int g = TT(f,ei);
            // If it is a boundary edge
            if (g == -1) continue;
            // If the opposite face is singular skip it
            if (regular_max.find(g) == regular_max.end()) continue;
            // If the opposite face is not marked
            if (!faces_marked_max[g]) continue;

            double e1_max = abs(e_max(F(f, ei))) < epsilon ? 0 : e_max(F(f, ei));
            double e2_max = abs(e_max(F(f, (ei+1)%3))) < epsilon ? 0 : e_max(F(f, (ei+1)%3));
            double e3_max = abs(e_max(F(f, (ei+2)%3))) < epsilon ? 0 : e_max(F(f, (ei+2)%3));

            // If there is no zero level set on the edge
            if ((e1_max > 0 && e2_max > 0) || (e1_max < 0 && e2_max < 0))
                continue;

            // The zero set range (p1, p2], excluding p1
            Eigen::RowVector3d p1 = V.row(F(f, ei));
            Eigen::RowVector3d p2 = V.row(F(f, (ei+1)%3));

            if (e2_max == 0) {
                zero_set.push_back(p2);
            }
            else {
                // If the obtained point is not this point
                if (e1_max != 0) {
                    Eigen::RowVector3d p0 = (e1_max * p2 - e2_max * p1) / (e1_max - e2_max);
                    zero_set.push_back(p0);
                }
            }
        }

        if (zero_set.size() == 2) {
            Eigen::MatrixXd edge(2, 3);
            edge.row(0) = zero_set[0];
            edge.row(1) = zero_set[1];
            edges_max.push_back(edge);
        }
        else if (zero_set.size() == 3) {
            for (int i = 0; i < 3; i++) {
                Eigen::MatrixXd edge(2, 3);
                edge.row(0) = B.row(f);
                edge.row(1) = zero_set[i];
                edges_max.push_back(edge);
            }
        }
    }

    // For e_min
    for (int f = 0; f < F.rows(); f++) {
        // If the triangle is regular, skip it
        if (regular_min.find(f) != regular_min.end()) continue;

        // Determine if any point on one edge is zero set
        vector<Eigen::RowVector3d> zero_set;
        for (int ei = 0; ei < F.cols(); ei++) {
            // Look up the opposite face
            int g = TT(f,ei);
            // If it is a boundary edge
            if (g == -1) continue;
            // If the opposite face is singular skip it
            if (regular_min.find(g) == regular_min.end()) continue;
            // If the opposite face is not marked
            if (!faces_marked_min[g]) continue;

            double e1_min = abs(e_min(F(f, ei))) < epsilon ? 0 : e_min(F(f, ei));
            double e2_min = abs(e_min(F(f, (ei+1)%3))) < epsilon ? 0 : e_min(F(f, (ei+1)%3));
            double e3_min = abs(e_min(F(f, (ei+2)%3))) < epsilon ? 0 : e_min(F(f, (ei+2)%3));

            // If there is no zero level set on the edge
            if ((e1_min > 0 && e2_min > 0) || (e1_min < 0 && e2_min < 0))
                continue;

            // The zero set range (p1, p2], excluding p1
            Eigen::RowVector3d p1 = V.row(F(f, ei));
            Eigen::RowVector3d p2 = V.row(F(f, (ei+1)%3));

            if (e2_min == 0) {
                zero_set.push_back(p2);
            }
            else {
                // If the obtained point is not this point
                if (e1_min != 0) {
                    Eigen::RowVector3d p0 = (e1_min * p2 - e2_min * p1) / (e1_min - e2_min);
                    zero_set.push_back(p0);
                }
            }
        }

        if (zero_set.size() == 2) {
            Eigen::MatrixXd edge(2, 3);
            edge.row(0) = zero_set[0];
            edge.row(1) = zero_set[1];
            edges_min.push_back(edge);
        }
        else if (zero_set.size() == 3) {
            for (int i = 0; i < 3; i++) {
                Eigen::MatrixXd edge(2, 3);
                edge.row(0) = B.row(f);
                edge.row(1) = zero_set[i];
                edges_min.push_back(edge);
            }
        }
    }
}

void smooth_feature_line(
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& k_vec_max,
    Eigen::MatrixXd& k_vec_min,
    Eigen::VectorXd& e_max,
    Eigen::VectorXd& e_min
    )
{
    double lambda = 0.1;
    Eigen::VectorXd delta_max = Eigen::MatrixXd::Constant(V.rows(), 1, 0);
    Eigen::VectorXd delta_min = Eigen::MatrixXd::Constant(V.rows(), 1, 0);

    for (int p = 0; p < V.rows(); p++) {
        vector<int> link = VV[p];

        for (int q = 0; q < link.size(); q++) {
            double weight = cotan_weight(p, link[q]);

            int sigma_max = k_vec_max.row(p).dot(k_vec_max.row(link[q])) < 0 ? -1 : 1;
            int sigma_min = k_vec_min.row(p).dot(k_vec_min.row(link[q])) < 0 ? -1 : 1;

            delta_max(p) += weight * (sigma_max*e_max(link[q]) - e_max(p));
            delta_min(p) += weight * (sigma_max*e_min(link[q]) - e_min(p));
        }
    }

    // Smooth the e
    e_max = e_max + lambda * delta_max;
    e_min = e_min + lambda * delta_min;
}

void plot_feature_lines(igl::opengl::glfw::Viewer& viewer, vector<Eigen::MatrixXd> edges_max, vector<Eigen::MatrixXd> edges_min) {
    // Plot the mesh
    viewer.data().clear();
    viewer.data().set_mesh(V, F);

    for (int i = 0; i < edges_max.size(); i++) {
        viewer.data().add_edges(edges_max[i].row(0), edges_max[i].row(1), Eigen::RowVector3d(1,0,0));
    }
    for (int i = 0; i < edges_min.size(); i++) {
        viewer.data().add_edges(edges_min[i].row(0), edges_min[i].row(1), Eigen::RowVector3d(0,0,1));
    }

    Eigen::MatrixXd C(F.rows(), 3);
    for (int i = 0; i < F.rows(); i++) {
        if (regular_max.find(i) == regular_max.end()) {
            C.row(i) << 0, 0, 0;
        }
        else
            C.row(i) << 1, 1, 1;
    }

    viewer.data().set_colors(C);

    // Disable wireframe
    viewer.data().show_lines = false;
}

// It allows to change the degree of the field when a number is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    using namespace Eigen;
    using namespace std;

    if (key == '1')
    {
        compute_extremality(V, F, TT, N_faces, N_vertices, k_max, k_min, k_vec_max, k_vec_min, e_max, e_min);
        feature_lines(V, F, k_max, k_min, k_vec_max, k_vec_min, e_max, e_min);
        plot_feature_lines(viewer, edges_max, edges_min);
    }

    if (key == '2')
    {
        smooth_feature_line(V, F, k_vec_max, k_vec_min, e_max, e_min);
        feature_lines(V, F, k_max, k_min, k_vec_max, k_vec_min, e_max, e_min);
        plot_feature_lines(viewer, edges_max, edges_min);
    }

    return false;
}

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace Eigen;

    if (argc != 2) {
        cout << "Usage final_project_bin mesh.off" << endl;
        exit(0);
    }

    // Load a mesh in OFF format
    igl::readOFF(argv[1], V, F);

    // Compute per-face normals
    igl::per_face_normals(V,F,N_faces);

    // Compute per-vertex normals
    igl::per_vertex_normals(V,F,N_vertices);

    // Triangle-triangle adjacency
    igl::triangle_triangle_adjacency(F,TT,TTi);

    // Vertex-triangle adjacency
    igl::vertex_triangle_adjacency(V,F,VT,VTi);

    // Compute area for each triangle
    igl::doublearea(V,F,A);

    // Compute the bary center for each triangle
    igl::barycenter(V,F,B);

    // Compute the aadjacency list for each vertex
    igl::adjacency_list(F,VV);

    // Compute gradient operator
    igl::grad(V,F,G);

    // Interpolate the field and plot
    key_down(viewer, '1', 0);

    // Register the callbacks
    viewer.callback_key_down = &key_down;

    // Disable wireframe
    viewer.data().show_lines = false;

    // Launch the viewer
    viewer.launch();

    return 0;
}
