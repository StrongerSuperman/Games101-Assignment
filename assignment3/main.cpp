#include <iostream>
#include <opencv2/opencv.hpp>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

#define BLINN_PHONG

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // TODO: Use the same projection matrix from the previous assignments
    Eigen::Matrix4f projection;
    float half_fov_rad = eye_fov / 2.0f * MY_PI / 180.0f;
    float tan_half_fov = tan(half_fov_rad);
    projection << 1.0f/tan_half_fov*aspect_ratio, 0, 0, 0,
                  0, 1.0f/tan_half_fov, 0, 0,
                  0, 0, (zFar+zNear)/(zNear-zFar), (2.0f*zFar*zNear)/(zNear-zFar),
                  0, 0, -1, 1;

    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};


Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f texture_color;
    if (payload.texture)
    {
        texture_color = payload.texture->getColorRGB(payload.tex_coords.x(), payload.tex_coords.y());
    }

    auto l1 = light{ {20, 20, 20}, {0.5f, 0.5f, 0.5f} };
    auto l2 = light{ {-20, 20, 20},  {0.5f, 0.5f, 0.5f} };

    std::vector<light> lights = { l1, l2 };
    Eigen::Vector3f amb_light_intensity{0.05, 0.05, 0.05 };
    Eigen::Vector3f eye_pos{0, 0, 10};

    Eigen::Vector3f color = texture_color / 255.f;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // ambient
        result_color += Eigen::Vector3f(amb_light_intensity.array() * color.array());
        // diffuse
        auto light_dir = (light.position - point).normalized();
        auto diff_factor = MAX(0, light_dir.dot(normal));
        result_color += Eigen::Vector3f(diff_factor * light.intensity.array() * color.array());
        // specular
#ifdef BLINN_PHONG
        // use half vector dot nomal
        auto half_vec = (eye_pos - point + light_dir).normalized();
        auto spec_factor = pow(MAX(0, half_vec.dot(normal)), 32.0f);
#else
        // use reflect vector dot view vector
        auto view_dir = (eye_pos - point).normalized();
        auto reflect_vec = reflect(light_dir, normal);
        auto spec_factor = pow(MAX(0, reflect_vec.dot(view_dir)), 8.0f);
#endif
        result_color += Eigen::Vector3f(spec_factor * light.intensity.array() * color.array());
    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    auto l1 = light{{20, 20, 20}, {0.5f, 0.5f, 0.5f}};
    auto l2 = light{{-20, 20, 0},  {0.5f, 0.5f, 0.5f} };

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{ 0.05, 0.05, 0.05 };
    Eigen::Vector3f eye_pos{0, 0, 10};

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        // ambient
        result_color += Eigen::Vector3f(amb_light_intensity.array() * color.array());
        // diffuse
        auto light_dir = (light.position - point).normalized();
        auto diff_factor = MAX(0, light_dir.dot(normal));
        result_color += Eigen::Vector3f(diff_factor * light.intensity.array() * color.array());
        // specular
#ifdef BLINN_PHONG
        // use half vector dot nomal
        auto half_vec = (eye_pos - point + light_dir).normalized();
        auto spec_factor = pow(MAX(0, half_vec.dot(normal)), 32.0f);
#else
        // use reflect vector dot view vector
        auto view_dir = (eye_pos - point).normalized();
        auto reflect_vec = reflect(light_dir, normal);
        auto spec_factor = pow(MAX(0, reflect_vec.dot(view_dir)), 8.0f);
#endif
        result_color += Eigen::Vector3f(spec_factor * light.intensity.array() * color.array());
    }
    return result_color * 255.f;
}



Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    auto l1 = light{ {20, 20, 20}, {0.5f, 0.5f, 0.5f} };
    auto l2 = light{ {-20, 20, 0},  {0.5f, 0.5f, 0.5f} };

    std::vector<light> lights = { l1, l2 };
    Eigen::Vector3f amb_light_intensity{ 0.05, 0.05, 0.05 };
    Eigen::Vector3f eye_pos{ 0, 0, 10 };

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;

    auto n = normal;
    auto t = Eigen::Vector3f(
        n.x() * n.y() / sqrt(n.x() * n.x() + n.z() * n.z()),
        sqrt(n.x() * n.x() + n.z() * n.z()),
        n.z() * n.y() / sqrt(n.x() * n.x() + n.z() * n.z())
    );
    auto b = n.cross(t);
    Eigen::Matrix3f tbn;
    tbn << t, b, n;
    auto hmap = payload.texture;
    auto u = payload.tex_coords.x();
    auto v = payload.tex_coords.y();
    auto du = kh * kn * (hmap->getColorGray(u + 1.0f / hmap->width, v) - hmap->getColorGray(u, v));
    auto dv = kh * kn * (hmap->getColorGray(u, v + 1.0f / hmap->height) - hmap->getColorGray(u, v));
    Eigen::Vector3f perturb_n;
    perturb_n << -du, -dv, 1;
    perturb_n.normalize();
    normal = (tbn * perturb_n).normalized();
    point = point + kn * normal * hmap->getColorGray(u, v);

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // ambient
        result_color += Eigen::Vector3f(amb_light_intensity.array() * color.array());
        // diffuse
        auto light_dir = (light.position - point).normalized();
        auto diff_factor = MAX(0, light_dir.dot(normal));
        result_color += Eigen::Vector3f(diff_factor * light.intensity.array() * color.array());
        // specular
#ifdef BLINN_PHONG
        // use half vector dot nomal
        auto half_vec = (eye_pos - point + light_dir).normalized();
        auto spec_factor = pow(MAX(0, half_vec.dot(normal)), 32.0f);
#else
        // use reflect vector dot view vector
        auto view_dir = (eye_pos - point).normalized();
        auto reflect_vec = reflect(light_dir, normal);
        auto spec_factor = pow(MAX(0, reflect_vec.dot(view_dir)), 8.0f);
#endif
        result_color += Eigen::Vector3f(spec_factor * light.intensity.array() * color.array());
    }

    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    auto l1 = light{ {20, 20, 20}, {0.5f, 0.5f, 0.5f} };
    auto l2 = light{ {-20, 20, 0},  {0.5f, 0.5f, 0.5f} };

    std::vector<light> lights = { l1, l2 };
    Eigen::Vector3f amb_light_intensity{ 0.05, 0.05, 0.05 };
    Eigen::Vector3f eye_pos{ 0, 0, 10 };

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;

    auto n = normal;
    auto t = Eigen::Vector3f(
        n.x() * n.y() / sqrt(n.x() * n.x() + n.z() * n.z()),
        sqrt(n.x() * n.x() + n.z() * n.z()),
        n.z() * n.y() / sqrt(n.x() * n.x() + n.z() * n.z())
    );
    auto b = n.cross(t);
    Eigen::Matrix3f tbn;
    tbn << t, b, n;
    auto hmap = payload.texture;
    auto u = payload.tex_coords.x();
    auto v = payload.tex_coords.y();
    auto du = kh * kn * (hmap->getColorGray(u + 1.0f / hmap->width, v) - hmap->getColorGray(u, v));
    auto dv = kh * kn * (hmap->getColorGray(u, v + 1.0f / hmap->height) - hmap->getColorGray(u, v));
    Eigen::Vector3f perturb_n;
    perturb_n << -du, -dv, 1;
    perturb_n.normalize();
    normal = (tbn * perturb_n).normalized();

    Eigen::Vector3f result_color = normal;

    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "../models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader;

    //active_shader = normal_fragment_shader;

    //active_shader = phong_fragment_shader;

    //auto texture_path = "spot_texture.png";
    //r.set_texture(Texture(obj_path + texture_path));
    //active_shader = texture_fragment_shader;

    //auto texture_path = "hmap.jpg";
    //r.set_texture(Texture(obj_path + texture_path));
    //active_shader = bump_fragment_shader;

    auto texture_path = "hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));
    active_shader = displacement_fragment_shader;
    
    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}
