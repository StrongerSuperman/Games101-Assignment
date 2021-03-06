// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return { c1,c2,c3 };
}

static bool insideTriangle(int x, int y, const Vector3f* _v)
{   
    Vector2f p = Vector2f(x, y);
    Vector2f a = Vector2f(_v[0].x(), _v[0].y());
    Vector2f b = Vector2f(_v[1].x(), _v[1].y());
    Vector2f c = Vector2f(_v[2].x(), _v[2].y());

    // p3p1 x p3p2
    auto crossProduct = [](Vector2f& p1, Vector2f& p2, Vector2f& p3) {
        return (p1.x() - p3.x()) * (p2.y() - p3.y()) - (p2.x() - p3.x()) * (p1.y() - p3.y());
    };

#define SLN_3_OPTM

#ifdef SLN_1
    // sum of trangle's area
    float areaAll = 0.5f * abs(crossProduct(a, b, c));
    float area1 = 0.5f * abs(crossProduct(p, a, b));
    float area2 = 0.5f * abs(crossProduct(p, b, c));
    float area3 = 0.5f * abs(crossProduct(p, c, a));
    if (area1 == 0 || area2 == 0 || area3 == 0) return false;
    return abs(area1 + area2 + area3 - areaAll) < 0.01f;  // 0.01f is decided by float precision error
#elif defined(SLN_2)
    /* barycentric coordinates in triangle
     * solve the equation: p = p0 + (p1 - p0) * s + (p2 - p0) * t
     * s,t and 1 - s - t are called the barycentric coordinates of the point p
     */
    auto [s, t, w] = computeBarycentric2D(x, y, _v);
    if (s < 0 || s > 1) return false;
    if (t < 0 || t > 1) return false;
    return u + v <= 1;
#elif defined(SLN_3)
    /* point in same side of CW/CCW triangle's three sides
     * check the sign of three crossproducts between triangle's three side and vector of point to triangle's three vertex
     * reference: https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
     */
    float d1, d2, d3;
    bool has_neg, has_pos;
    d1 = crossProduct(p, a, b);
    d2 = crossProduct(p, b, c);
    d3 = crossProduct(p, c, a);
    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(has_neg && has_pos);
#elif defined(SLN_3_OPTM)
    // optimization for point in same side of CW/CCW triangle's three sides
    int ap_x = p.x() - a.x();
    int ap_y = p.y() - a.y();
    bool p_ab = (b.x() - a.x()) * ap_y - (b.y() - a.y()) * ap_x > 0;
    if ((c.x() - a.x()) * ap_y - (c.y() - a.y()) * ap_x > 0 == p_ab) return false;
    if ((c.x() - b.x()) * (p.y() - b.y()) - (c.y() - b.y()) * (p.x() - b.x()) > 0 != p_ab) return false;
    return true;
#endif
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();

    auto find_bbox = [&v]()->std::tuple<float, float, float, float> {
        float max_x = v[0].x();
        float min_x = v[0].x();
        float max_y = v[0].y();
        float min_y = v[0].y();
        for (int i = 1; i < 3; i++) {
            float& x = v[i].x();
            float& y = v[i].y();
            if (x > max_x) { max_x = x; }
            else if (x < min_x) { min_x = x; }
            if (y > max_y) { max_y = y; }
            else if (y < min_y) { min_y = y; }
        }
        return { max_x, min_x, max_y, min_y };
    };
    auto [bb_x_max, bb_x_min, bb_y_max, bb_y_min] = find_bbox();
    for (int y = bb_y_min; y <= bb_y_max; y++)
    {
        for (int x = bb_x_min; x <= bb_x_max; x++)
        {
            if (!insideTriangle(x, y, t.v))
                continue;
            auto [alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
            float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
            z_interpolated *= w_reciprocal;
            auto ind = (height-1-y)*width + x;
            // early-z
            if(z_interpolated < depth_buf[ind])
            {
                set_pixel(Eigen::Vector3f(x, y, z_interpolated), t.getColor());
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;
    depth_buf[ind] = point.z();
}

// clang-format on