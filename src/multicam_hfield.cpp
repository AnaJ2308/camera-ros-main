// g++/CMake with: up to 3 cameras, with transformations, mujoco, glfw - slow
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

// =========================== CONFIG ===========================
static const std::vector<std::string> CAM_TOPICS = {
    "/topic_1/cam_1/depth/color/points",
    "/topic_2/cam_2/depth/color/points",
    "/topic_3/cam_3/depth/color/points",
    // "camera/camera/depth/color/points"
};

static const char* TARGET_FRAME = "base_link";

// Heightfield grid & world size
static constexpr int   NROW = 160;
static constexpr int   NCOL = 160;
static constexpr double Lx  = 6.0;   // meters (world X span)
static constexpr double Ly  = 6.0;   // meters (world Y span)
static constexpr double Hz  = 0.8;   // vertical exaggeration (MuJoCo hfield size[2])
static constexpr double base = 0.05; // MuJoCo hfield size[3] base offset

static std::mutex g_mj_mutex; 
static std::atomic<bool> g_renderer_ready{false};

// #include <atomic>

// static std::atomic<uint64_t> g_cloud_msgs{0};
// static std::atomic<uint64_t> g_points_seen{0};
// static std::atomic<uint64_t> g_uploads{0};
// static std::atomic<uint64_t> g_renders{0};

static std::atomic<bool> g_pending_upload{false};



// ROI in TARGET_FRAME coordinates (mapped directly to grid)
static constexpr double ROI_X_HALF = Lx / 2.0;
static constexpr double ROI_Y_HALF = Ly / 2.0;

// z normalization bounds when pushing to MuJoCo
static constexpr float Z_MIN = 0.00f;
static constexpr float Z_MAX = 1.00f;

// Grid resolution
static constexpr double dx = (2.0 * ROI_X_HALF) / static_cast<double>(NCOL);
static constexpr double dy = (2.0 * ROI_Y_HALF) / static_cast<double>(NROW);

// ========================= MuJoCo Helpers =========================
static std::string make_model_xml(int nrow=NROW, int ncol=NCOL,
                                  double Lx_=Lx, double Ly_=Ly, double Hz_=Hz, double base_=base)
{
    char buf[4096];
    std::snprintf(buf, sizeof(buf),
R"XML(<?xml version="1.0"?>
<mujoco model="pc2_live_hfield">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <asset>
    <hfield name="terrain" nrow="%d" ncol="%d" size="%.4f %.4f %.4f %.4f"/>
  </asset>
  <worldbody>
    <light name="toplight" pos="0 0 5" dir="0 0 -1" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3" directional="true"/>
    <geom type="hfield" hfield="terrain" rgba="0.7 0.7 0.7 1"/>
    <camera name="iso" pos="2.0 -2.5 1.8" euler="35 0 25"/>
  </worldbody>
</mujoco>
)XML", nrow, ncol, Lx_, Ly_, Hz_, base_);
    return std::string(buf);
}

// Write XML to a temp file because mj_loadXML expects a file (simplest portable path)
static std::string write_temp_xml(const std::string& xml)
{
    std::string path = "/tmp/multicam_hfield.xml";
    std::ofstream f(path, std::ios::binary);
    f << xml;
    return path;
}

static void set_heightfield(mjModel* m, int hid, const float* arr01 /*size NROW*NCOL*/)
{
    if (!m || hid < 0 || hid >= m->nhfield) return;
    int nrow = m->hfield_nrow[hid];
    int ncol = m->hfield_ncol[hid];
    int adr  = m->hfield_adr[hid];
    if (nrow <= 0 || ncol <= 0) return;
    std::memcpy(m->hfield_data + adr, arr01, sizeof(float) * nrow * ncol);
}

static void upload_heightfield(mjModel* m, int hid, const mjrContext* con)
{
    // mjr_uploadHField requires a valid rendering context
    if (!m || !con || hid < 0 || hid >= m->nhfield) return;
    mjr_uploadHField(m, con, hid);
}

// ========================= Fusion Helpers =========================
struct GridState {
    // Max-fused meters grid, persistent
    std::vector<float> grid_m;     // size NROW*NCOL, initialized to -inf
    std::vector<float> heights01;  // normalized buffer [0,1] for MuJoCo upload
    std::mutex mtx;
    bool new_frame = false;

    GridState()
    : grid_m(NROW*NCOL, -INFINITY),
      heights01(NROW*NCOL, 0.0f)
    {}
};

// Max-fuse a transformed cloud (already in TARGET_FRAME)
static inline void fuse_points_to_grid_meters(
    const sensor_msgs::msg::PointCloud2& cloud_tf,
    GridState& gs
){
    // Iterators over x,y,z fields (float32)
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud_tf, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud_tf, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud_tf, "z");

    const int width  = static_cast<int>(cloud_tf.width);
    const int height = static_cast<int>(cloud_tf.height);
    if (width == 0 || height == 0) return;

    auto& g = gs.grid_m; // flattened NROW*NCOL

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        float x = *iter_x;
        float y = *iter_y;
        float z = *iter_z + 1.0f;   // NOTE: mirrors your +1.0 offset

        if (!std::isfinite(z)) continue;
        if (x < -ROI_X_HALF || x >= ROI_X_HALF) continue;
        if (y < -ROI_Y_HALF || y >= ROI_Y_HALF) continue;

        int col = static_cast<int>((x + ROI_X_HALF) / dx);
        int row = static_cast<int>((y + ROI_Y_HALF) / dy);
        if (col < 0) col = 0; else if (col >= NCOL) col = NCOL-1;
        if (row < 0) row = 0; else if (row >= NROW) row = NROW-1;

        int lin = row * NCOL + col;

        // manual max (avoids atomic; single-threaded in callback under lock)
        float& cell = g[lin];
        if (z > cell) cell = z;
    }
}

// ========================= ROS2 Node =========================
class MultiCamHFieldNode : public rclcpp::Node {
public:
    MultiCamHFieldNode(
        mjModel* model, mjData* data, int hid,
        mjvScene* scene, mjvCamera* cam, mjrContext* con, GLFWwindow* window, std::mutex* mj_mutex)
    : Node("multicam_pc2_to_hfield"),
      model_(model), data_(data), hfield_id_(hid),
      scene_(scene), cam_(cam), con_(con), window_(window), mj_mutex_(mj_mutex),
      tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        // Subscriptions (sensor data QoS)
        
        rclcpp::QoS qos = rclcpp::SensorDataQoS();

        for (const auto& topic : CAM_TOPICS) {
            auto sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                topic, qos,
                [this, topic](const sensor_msgs::msg::PointCloud2::SharedPtr msg){
                    this->cloud_callback(topic, msg);
                });
            subs_.push_back(sub);
        }
        for (const auto& topic : CAM_TOPICS) {
            RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", topic.c_str());
        }


        RCLCPP_INFO(this->get_logger(),
            "Fusing %zu cameras into %s. ROI=(%.2fm x %.2fm), grid=%dx%d",
            CAM_TOPICS.size(), TARGET_FRAME, Lx, Ly, NROW, NCOL);

        // 10 Hz timer to normalize+upload+clear
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(400),
            std::bind(&MultiCamHFieldNode::on_tick, this));
    }

    GridState& grid() { return grid_; }

private:
    void cloud_callback(const std::string& topic, const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // TF lookup at message time
        size_t n_raw = static_cast<size_t>(msg->width) * msg->height;
        // g_points_seen += n_raw;
        // g_cloud_msgs++;
        // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        //                     "[%s] received cloud: %zu pts (stamp=%.3f)",
        //                     topic.c_str(), n_raw, rclcpp::Time(msg->header.stamp).seconds());

        geometry_msgs::msg::TransformStamped tf;
        try {
            tf = tf_buffer_.lookupTransform(
                TARGET_FRAME, msg->header.frame_id,
                msg->header.stamp,
                std::chrono::milliseconds(50));
        } catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "[%s] TF %s <- %s failed: %s",
                topic.c_str(), TARGET_FRAME, msg->header.frame_id.c_str(), e.what());
            return;
        }

        sensor_msgs::msg::PointCloud2 cloud_tf;
        try {
            tf2::doTransform(*msg, cloud_tf, tf);
            RCLCPP_DEBUG(get_logger(), "[%s] TF->doTransform OK", topic.c_str());

        } catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "[%s] doTransform failed: %s", topic.c_str(), e.what());
            return;
        }

        std::lock_guard<std::mutex> lk(grid_.mtx);
        fuse_points_to_grid_meters(cloud_tf, grid_);
        grid_.new_frame = true;
    }

    void on_tick()
    {
        
        std::vector<float> grid_copy;
        {
            std::lock_guard<std::mutex> lk(grid_.mtx);
            if (!grid_.new_frame) return;
            grid_copy = grid_.grid_m;   // copy meters grid
            grid_.new_frame = false;
        }

        // Replace -inf with Z_MIN
        for (auto& v : grid_copy) {
            if (!std::isfinite(v)) v = Z_MIN;
        }

        // Normalize to [0,1]
        float denom = std::max(1e-6f, (Z_MAX - Z_MIN));
        for (size_t i = 0; i < grid_copy.size(); ++i) {
            float x = (grid_copy[i] - Z_MIN) / denom;
            if (x < 0.f) x = 0.f; else if (x > 1.f) x = 1.f;
            grid_copy[i] = x;
        }

        {
            std::lock_guard<std::mutex> lk(grid_.mtx);
            // copy normalized into upload buffer
            std::memcpy(grid_.heights01.data(), grid_copy.data(),
                        grid_copy.size() * sizeof(float));

        }
        if (!g_renderer_ready.load(std::memory_order_acquire)) return;
        {
            std::lock_guard<std::mutex> lk(*mj_mutex_);
            // upload to MuJoCo
            set_heightfield(model_, hfield_id_, grid_.heights01.data());
            // upload_heightfield(model_, hfield_id_, con_);
            g_pending_upload.store(true, std::memory_order_release);
        }    

            
        {
            std::lock_guard<std::mutex> lk(grid_.mtx);
            // clear for next round
            std::fill(grid_.grid_m.begin(), grid_.grid_m.end(), -INFINITY);
        }

    }

private:
    // MuJoCo
    mjModel* model_;
    mjData*  data_;
    int      hfield_id_;
    mjvScene*   scene_;
    mjvCamera*  cam_;
    mjrContext* con_;
    GLFWwindow* window_;
    std::mutex* mj_mutex_;

    // ROS
    std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subs_;
    rclcpp::TimerBase::SharedPtr timer_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Shared grid
    GridState grid_;
};

// ========================= GLFW + MuJoCo Viewer =========================
static void mj_error_fn(int error, const char* msg)
{
    std::cerr << "MuJoCo error " << error << ": " << (msg?msg:"") << std::endl;
}

int main(int argc, char** argv)
{
    

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(1280, 800, "Multi-Cam Heightfield (ROS2 + MuJoCo)", nullptr, nullptr);
    if (!window) { std::cerr << "Failed to create GLFW window\n"; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Load model from XML string
    std::string xml = make_model_xml();
    std::string path = write_temp_xml(xml);
    char error[1024] = {0};
    mjModel* m = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
    if (!m) {
        std::cerr << "mj_loadXML error: " << error << "\n";
        glfwTerminate();
        return 1;
    }
    mjData* d = mj_makeData(m);

    // Visualization structures
    mjvCamera cam;
    mjv_defaultCamera(&cam);
    // match your python viewer feel a bit
    cam.type = mjCAMERA_FREE;
    
    cam.lookat[0] = 0;  cam.lookat[1] = 0;  cam.lookat[2] = 0.5; // target
    cam.distance   = 10.5;   // bigger = farther = zoomed out
    cam.azimuth    = 100;    // deg around z
    cam.elevation  = -30;   // deg from horizontal

    mjvOption opt; mjv_defaultOption(&opt);
    mjvScene scn;  mjv_defaultScene(&scn);
    mjrContext con; mjr_defaultContext(&con);

    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);
    g_renderer_ready.store(true, std::memory_order_release);


 

    int hid = mj_name2id(m, mjOBJ_HFIELD, "terrain");
    if (hid < 0) {
        std::cerr << "ERROR: hfield 'terrain' not found in model.\n";
        rclcpp::shutdown();
        mj_deleteData(d);
        mj_deleteModel(m);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    
    {
        std::lock_guard<std::mutex> lk(g_mj_mutex);
        
        upload_heightfield(m, hid, &con);
    }

    // --- ROS2 init ---
    rclcpp::init(argc, argv);
    // Node needs MuJoCo handles for upload & a timer callback
    auto node = std::make_shared<MultiCamHFieldNode>(m, d, hid, &scn, &cam, &con, window, &g_mj_mutex);

    // Spin ROS on a background thread
    std::thread ros_thread([&](){
        rclcpp::spin(node);
    });

    // Render loop
    
    while (!glfwWindowShouldClose(window)) {
       
        {
            std::lock_guard<std::mutex> lk(g_mj_mutex);
            // Step the physics
            mj_step(m, d);
            // Update the scene
            mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        }
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (width <= 0 || height <= 0) {
            glfwPollEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        static int last_w = 0, last_h = 0;
        {
            std::lock_guard<std::mutex> lk(g_mj_mutex);
            if (width != last_w || height != last_h) {
                mjr_restoreBuffer(&con);
                last_w = width;
                last_h = height;
            }
            
        }
       
        mjrRect viewport{0, 0, width, height};
        {
            std::lock_guard<std::mutex> lk(g_mj_mutex);
            if (g_pending_upload.load(std::memory_order_acquire)) {
                upload_heightfield(m, hid, &con);         // now on the GL thread → safe
            g_pending_upload.store(false, std::memory_order_release);
        }
            mjr_render(viewport, &scn, &con);
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    rclcpp::shutdown();
    if (ros_thread.joinable()) ros_thread.join();

    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(d);
    mj_deleteModel(m);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
