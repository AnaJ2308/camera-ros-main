// g++/CMake with: rclcpp sensor_msgs MuJoCo GLFW
// Single-camera → hardcoded TF, R and t -> MuJoCo (no TF lookup) - very fast
// ROS2 + MuJoCo live heightfield viewer

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using std::placeholders::_1;

// =========================== CONFIG ===========================
static const char* TOPIC = "/camera/camera/depth/color/points";

constexpr int   NROW = 160;
constexpr int   NCOL = 160;
constexpr float Lx   = 6.0f;
constexpr float Ly   = 6.0f;
constexpr float Hz   = 0.8f;
constexpr float base = 0.05f;

constexpr float ROI_X_HALF = Lx * 0.5f;
constexpr float ROI_Y_HALF = Ly * 0.5f;

constexpr float Z_MIN = -1.0f;
constexpr float Z_MAX =  1.0f;

constexpr float dx = (2.0f * ROI_X_HALF) / NCOL;
constexpr float dy = (2.0f * ROI_Y_HALF) / NROW;

// Hardcoded rotation (R) and translation (t)
static const float R_[3][3] = {
    {  0.0f,        -0.70703739f,  0.70717621f },
    { -1.0f,        -0.0f,         0.0f        },
    { -1.110223e-16f, -0.70717621f,-0.70703739f }
};
static const float t_[3] = { 0.0988f, 0.0f, 0.028f };

// =============================================================

// Minimal MuJoCo viewer wrapper
struct MjViewer {
  GLFWwindow* window = nullptr;
  mjModel*    m = nullptr;
  mjData*     d = nullptr;
  mjvCamera   cam;
  mjvOption   opt;
  mjvScene    scn;
  mjrContext  con;
  int         hid = -1;

  bool init(const std::string& xml_path) {
    char error[1024] = {0};
    m = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    if (!m) {
      std::cerr << "mj_loadXML error:\n" << error << std::endl;
      return false;
    }
    d = mj_makeData(m);

    if (!glfwInit()) {
      std::cerr << "glfwInit failed\n";
      return false;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    window = glfwCreateWindow(1280, 800, "onecam_mytf", nullptr, nullptr);
    if (!window) {
      std::cerr << "glfwCreateWindow failed\n";
      return false;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // camera similar to Python (iso)
    cam.type = mjCAMERA_FREE;
    cam.distance = 10.5;
    cam.azimuth  = 100;
    cam.elevation= -30;
    cam.lookat[0]= 0.0; cam.lookat[1]= 0.0; cam.lookat[2]= 0.5;

    // find hfield id
    hid = mj_name2id(m, mjOBJ_HFIELD, "terrain");
    if (hid < 0) {
      std::cerr << "Could not find hfield 'terrain'\n";
      return false;
    }
    return true;
  }

  void upload_hfield() {
    // upload hfield texture/mesh to GPU
    mjr_uploadHField(m, &con, hid);
  }

  bool should_close() const { return window && !glfwWindowShouldClose(window); }

  void step_and_render() {
    // step
    mj_step(m, d);

    // render
    mjrRect viewport = {0, 0, 0, 0};
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    viewport.width  = width;
    viewport.height = height;

    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  void close() {
    if (window) { glfwDestroyWindow(window); window = nullptr; }
    mjr_freeContext(&con);
    mjv_freeScene(&scn);
    if (d) { mj_deleteData(d); d = nullptr; }
    if (m) { mj_deleteModel(m); m = nullptr; }
    glfwTerminate();
  }
};

// Create MJCF as a string and write to a temp file
static std::string write_model_xml_to_tmp() {
  std::string xml;
  xml  = "<mujoco model='onecam_mytf'>\n";
  xml += "  <option timestep='0.005' gravity='0 0 -9.81'/>\n";
  xml += "  <asset>\n";
  xml += "    <hfield name='terrain' nrow='" + std::to_string(NROW) +
         "' ncol='" + std::to_string(NCOL) +
         "' size='" + std::to_string(Lx) + " " + std::to_string(Ly) + " " +
         std::to_string(Hz) + " " + std::to_string(base) + "'/>\n";
  xml += "  </asset>\n";
  xml += "  <worldbody>\n";
  xml += "    <light name='toplight' pos='0 0 5' dir='0 0 -1' diffuse='0.4 0.4 0.4' specular='0.3 0.3 0.3' directional='true'/>\n";
  xml += "    <geom type='hfield' hfield='terrain' rgba='0.7 0.7 0.7 1'/>\n";
  xml += "    <camera name='iso' pos='0 0 10' euler='-30 0 0'/>\n";
  xml += "  </worldbody>\n";
  xml += "</mujoco>\n";

  // write
  std::string path = "/tmp/onecam_mytf.xml";
  std::ofstream ofs(path, std::ios::binary);
  ofs << xml;
  ofs.close();
  return path;
}

// set_heightfield: heights01 must be NROW*NCOL floats, row-major
static void set_heightfield(mjModel* m, int hid, const std::vector<float>& heights01) {
  const int nrow = m->hfield_nrow[hid];
  const int ncol = m->hfield_ncol[hid];
  if ((int)heights01.size() != nrow * ncol) return;
  const int adr = m->hfield_adr[hid];
  std::memcpy(m->hfield_data + adr, heights01.data(), sizeof(float) * heights01.size());
}

// Node that subscribes and fills a shared heights buffer
class PC2ToHFieldNode final : public rclcpp::Node {
public:
  PC2ToHFieldNode(mjModel* model, int hfield_id,
                  std::vector<float>& heights01,
                  std::mutex& mtx,
                  std::atomic_bool& new_frame)
  : Node("pc2_to_hfield_onecam_tf_fast"),
    m_(model), hid_(hfield_id),
    heights01_(heights01), mtx_(mtx), new_frame_(new_frame) {

    auto qos = rclcpp::SensorDataQoS();
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      TOPIC, qos,
      std::bind(&PC2ToHFieldNode::callback, this, _1)
    );

    RCLCPP_INFO(this->get_logger(), "[ROS2] Subscribed to %s", TOPIC);
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Prepare a flat grid of Z values initialized to Z_MIN
    std::vector<float> maxZ(NROW * NCOL, Z_MIN);

    // Iterate x,y,z as float fields
    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");

    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
      const float x = *it_x;
      const float y = *it_y;
      const float z = *it_z;

      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        continue;

      // Transform: x' = R x + t
      float xb = R_[0][0]*x + R_[0][1]*y + R_[0][2]*z + t_[0];
      float yb = R_[1][0]*x + R_[1][1]*y + R_[1][2]*z + t_[1];
      float zb = R_[2][0]*x + R_[2][1]*y + R_[2][2]*z + t_[2];

      // ROI filter
      if (xb < -ROI_X_HALF || xb >= ROI_X_HALF) continue;
      if (yb < -ROI_Y_HALF || yb >= ROI_Y_HALF) continue;
      if (!std::isfinite(zb)) continue;

      // Bin to grid
      int col = static_cast<int>((xb + ROI_X_HALF) / dx);
      int row = static_cast<int>((yb + ROI_Y_HALF) / dy);
      if (col < 0) col = 0; if (col >= NCOL) col = NCOL - 1;
      if (row < 0) row = 0; if (row >= NROW) row = NROW - 1;

      const int lin = row * NCOL + col;
      if (zb > maxZ[lin]) maxZ[lin] = zb;
    }

    // Normalize to [0,1] and copy into shared heights01
    const float denom = std::max(1e-6f, (Z_MAX - Z_MIN));
    std::lock_guard<std::mutex> lk(mtx_);
    for (int i = 0; i < NROW * NCOL; ++i) {
      float v = maxZ[i];
      // If untouched cell stayed at Z_MIN, it maps to 0
      v = (v - Z_MIN) / denom;
      // No need to clip strictly; MuJoCo tolerates slight overs/unders
    //   if (v < 0.f) v = 0.f;
    //   if (v > 1.f) v = 1.f;
      heights01_[i] = v;
    }
    new_frame_.store(true, std::memory_order_release);
  }

  mjModel* m_;
  int hid_;
  std::vector<float>& heights01_;
  std::mutex& mtx_;
  std::atomic_bool& new_frame_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc, char** argv) {
  // 1) MuJoCo model/viewer
  const std::string xml_path = write_model_xml_to_tmp();
  MjViewer viewer;
  if (!viewer.init(xml_path)) {
    std::cerr << "Failed to init MuJoCo viewer.\n";
    return 1;
  }

  // heights buffer shared with ROS thread; row-major NROW*NCOL
  std::vector<float> heights01(NROW * NCOL, 0.0f);
  std::mutex heights_mtx;
  std::atomic_bool new_frame(false);

  // initial upload of hfield
  set_heightfield(viewer.m, viewer.hid, heights01);
  viewer.upload_hfield();

  // 2) ROS2
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PC2ToHFieldNode>(viewer.m, viewer.hid,
                                                heights01, heights_mtx, new_frame);

  std::thread ros_spin([&](){
    rclcpp::spin(node);
  });

  // 3) Main render loop
  using namespace std::chrono_literals;
  while (viewer.should_close()) {
    // If a new frame arrived, push HF data to model+GPU
    if (new_frame.load(std::memory_order_acquire)) {
      {
        std::lock_guard<std::mutex> lk(heights_mtx);
        set_heightfield(viewer.m, viewer.hid, heights01);
      }
      viewer.upload_hfield();
      new_frame.store(false, std::memory_order_release);
    }

    viewer.step_and_render();
    std::this_thread::sleep_for(2ms);
  }

  // 4) Cleanup
  rclcpp::shutdown();
  if (ros_spin.joinable()) ros_spin.join();
  viewer.close();
  return 0;
}
