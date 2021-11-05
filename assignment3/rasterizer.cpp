//
// Created by goksu on 4/6/19.
//

#include <algorithm>
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

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f>& normals)
{
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}


// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end)
{
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x,y,dx,dy,dx1,dy1,px,py,xe,ye,i;

    dx=x2-x1;
    dy=y2-y1;
    dx1=fabs(dx);
    dy1=fabs(dy);
    px=2*dy1-dx1;
    py=2*dx1-dy1;

    if(dy1<=dx1)
    {
        if(dx>=0)
        {
            x=x1;
            y=y1;
            xe=x2;
        }
        else
        {
            x=x2;
            y=y2;
            xe=x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point,line_color);
        for(i=0;x<xe;i++)
        {
            x=x+1;
            if(px<0)
            {
                px=px+2*dy1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    y=y+1;
                }
                else
                {
                    y=y-1;
                }
                px=px+2*(dy1-dx1);
            }
//            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point,line_color);
        }
    }
    else
    {
        if(dy>=0)
        {
            x=x1;
            y=y1;
            ye=y2;
        }
        else
        {
            x=x2;
            y=y2;
            ye=y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point,line_color);
        for(i=0;y<ye;i++)
        {
            y=y+1;
            if(py<=0)
            {
                py=py+2*dx1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    x=x+1;
                }
                else
                {
                    x=x-1;
                }
                py=py+2*(dx1-dy1);
            }
//            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point,line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(int x, int y, const Vector4f* _v){
    Vector3f v[3];
    for(int i=0;i<3;i++)
        v[i] = {_v[i].x(),_v[i].y(), 1.0};
    Vector3f f0,f1,f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x,y,1.);
    if((p.dot(f0)*f0.dot(v[2])>0) && (p.dot(f1)*f1.dot(v[0])>0) && (p.dot(f2)*f2.dot(v[1])>0))
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v){
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList) {

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (const auto& t:TriangleList)
    {
        Triangle newtri = *t;

        std::array<Eigen::Vector4f, 3> mm {
                (view * model * t->v[0]),
                (view * model * t->v[1]),
                (view * model * t->v[2])
        };

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto& v) {
            return v.template head<3>();
        });

        Eigen::Vector4f v[] = {
                mvp * t->v[0],
                mvp * t->v[1],
                mvp * t->v[2]
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec.x()/=vec.w();
            vec.y()/=vec.w();
            vec.z()/=vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {
                inv_trans * to_vec4(t->normal[0], 0.0f),
                inv_trans * to_vec4(t->normal[1], 0.0f),
                inv_trans * to_vec4(t->normal[2], 0.0f)
        };

        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            //screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            //view space normal
            newtri.setNormal(i, n[i].head<3>());
        }

        newtri.setColor(0, 148,121.0,92.0);
        newtri.setColor(1, 148,121.0,92.0);
        newtri.setColor(2, 148,121.0,92.0);

        // Also pass view space vertice position
        rasterize_triangle(newtri, viewspace_pos);
    }
}

static Eigen::Vector3f interpolate3(float weight[3], float recip_w[3], const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector3f& v3)
{
    float weight0 = weight[0] * recip_w[0];
    float weight1 = weight[1] * recip_w[1];
    float weight2 = weight[2] * recip_w[2];
    float normalizer = 1.0f / (weight0 + weight1 + weight2);
    float v_x = v1.x() * weight0 + v2.x() * weight1 + v3.x() * weight2;
    float v_y = v1.y() * weight0 + v2.y() * weight1 + v3.y() * weight2;
    float v_z = v1.z() * weight0 + v2.z() * weight1 + v3.z() * weight2;
    return Eigen::Vector3f(v_x, v_y, v_z) * normalizer;
}

static Eigen::Vector2f interpolate2(float weight[3], float recip_w[3], const Eigen::Vector2f& v1, const Eigen::Vector2f& v2, const Eigen::Vector2f& v3)
{
    float weight0 = weight[0] * recip_w[0];
    float weight1 = weight[1] * recip_w[1];
    float weight2 = weight[2] * recip_w[2];
    float normalizer = 1.0f / (weight0 + weight1 + weight2);
    float v_x = v1.x() * weight0 + v2.x() * weight1 + v3.x() * weight2;
    float v_y = v1.y() * weight0 + v2.y() * weight1 + v3.y() * weight2;
    return Eigen::Vector2f(v_x, v_y) * normalizer;
}

static float interpolate1(float weight[3], float recip_w[3], float v1, float v2, float v3)
{
    float weight0 = weight[0] * recip_w[0];
    float weight1 = weight[1] * recip_w[1];
    float weight2 = weight[2] * recip_w[2];
    float normalizer = 1.0f / (weight0 + weight1 + weight2);
    float v = v1 * weight0 + v2 * weight1 + v3 * weight2;
    return v * normalizer;
}

static void clamp2(Eigen::Vector2f& v1)
{
    auto clamp = [](float& v) {
        v = v < 0 ? 0 : v;
        v = v > 1 ? 1 : v;
    };
    clamp(v1.x());
    clamp(v1.y());
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos) 
{
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
            float weight[3] = { alpha, beta, gamma };
            float recip_w[3] = { 1.0 / t.v[0].w(), 1.0 / t.v[1].w(), 1.0 / t.v[2].w() };
            float z_interpolated = interpolate1(weight, recip_w, v[0].z(), v[1].z(), v[2].z());
            auto ind = (height-1-y)*width + x;
            // early-z
            if(z_interpolated < depth_buf[ind])
            {
                auto interpolated_color = interpolate3(weight, recip_w, t.color[0], t.color[1], t.color[2]);
                auto interpolated_normal = interpolate3(weight, recip_w, t.normal[0], t.normal[1], t.normal[2]);
                auto interpolated_texcoords = interpolate2(weight, recip_w, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2]);
                clamp2(interpolated_texcoords);
                fragment_shader_payload payload(
                    interpolated_color,
                    interpolated_normal.normalized(),
                    interpolated_texcoords,
                    texture ? &*texture : nullptr
                );
                payload.view_pos = interpolate3(weight, recip_w, view_pos[0], view_pos[1], view_pos[2]);
                auto pixel_color = fragment_shader(payload);
                set_pixel(Eigen::Vector2i(x, y), pixel_color);
                depth_buf[ind] = z_interpolated;
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

    texture = std::nullopt;
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-y)*width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color)
{
    //old index: auto ind = point.y() + point.x() * width;
    int ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}

