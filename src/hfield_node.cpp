// One camera, no transformations, takes the max
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <mujoco/mujoco.h>   // MuJoCo C API (v3+)
#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <limits>
#include <algorithm>
#include <fstream>
#include <string> 
#include <cstdio>
#include <unistd.h>  // getpid

#include <GLFW/glfw3.h>


// =========================== CONFIG (match your Python) ===========================
static constexpr int   NROW = 160;
static constexpr int   NCOL = 160;
static constexpr double Lx   = 6.0;   // meters covered in X
static constexpr double Ly   = 6.0;   // meters covered in Y
static constexpr double Hz   = 0.8;   // vertical scale for MuJoCo hfield
static constexpr double base = 0.05;  // base offset for MuJoCo hfield

// ROI in incoming pointcloud coords (x,y in meters): [-L/2, +L/2]
static constexpr double ROI_X_HALF = Lx * 0.5;
static constexpr double ROI_Y_HALF = Ly * 0.5;

// z→[0,1] mapping
static constexpr float Z_MIN = 0.0f;
static constexpr float Z_MAX = 1.0f;

static constexpr double dx = (2.0 * ROI_X_HALF) / double(NCOL);
static constexpr double dy = (2.0 * ROI_Y_HALF) / double(NROW);

// =========================== MuJoCo helpers ===========================
static std::string make_model_xml(int nrow, int ncol, double Lx_, double Ly_, double Hz_, double base_) {
  char buf[4096];
  // Simple ground + hfield + camera/light (camera can be used if you enable a viewer later)
  snprintf(buf, sizeof(buf),
R"(<?xml version="1.0"?>
<mujoco model="pc2_live_hfield">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <asset>
    <hfield name="terrain" nrow="%d" ncol="%d" size="%.3f %.3f %.3f %.3f"/>
  </asset>
  <worldbody>
    <light name="toplight" pos="0 0 5" dir="0 0 -1" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3" directional="true"/>
    <geom type="hfield" hfield="terrain" rgba="0.7 0.7 0.7 1"/>
    <camera name="iso" pos="0.0 0.0 0.0" euler="0 0 0"/>
  </worldbody>
</mujoco>
)", nrow, ncol, Lx_, Ly_, Hz_, base_);
  return std::string(buf);
}

static void set_heightfield(mjModel* m, int hid, const std::vector<float>& heights01) {
  const int nrow = m->hfield_nrow[hid];
  const int ncol = m->hfield_ncol[hid];
  if ((int)heights01.size() != nrow * ncol) return;
  const int adr = m->hfield_adr[hid];
  // MuJoCo expects row-major contiguous data
  for (int i = 0; i < nrow * ncol; ++i) {
    m->hfield_data[adr + i] = heights01[i];
  }
}

// =========================== Grid projection (max aggregation) ===========================
static void project_points_to_grid_max(
    const float* xs, const float* ys, const float* zs, size_t N,
    std::vector<float>& out_heights01)
{
  // flat buffer initialized to Z_MIN, we’ll take max into it
  std::fill(out_heights01.begin(), out_heights01.end(), Z_MIN);

  // Precompute useful constants
  const float x_half = static_cast<float>(ROI_X_HALF);
  const float y_half = static_cast<float>(ROI_Y_HALF);
  const float inv_dx = 1.0f / static_cast<float>(dx);
  const float inv_dy = 1.0f / static_cast<float>(dy);
  const float zmin   = Z_MIN;
  const float zmax   = Z_MAX;
  const float denom  = std::max(1e-6f, zmax - zmin);

  for (size_t i = 0; i < N; ++i) {
    float x = xs[i];
    float y = ys[i];
    float z = zs[i];

    // z transform (match your Python): z = -z + 1.0
    z = -z + 1.0f;

    // filter ROI & finite
    if (!(x >= -x_half && x < x_half && y >= -y_half && y < y_half))
      continue;
    if (!std::isfinite(z)) continue;

    // indices on the XY plane
    int col = static_cast<int>(std::floor((x + x_half) * inv_dx));
    int row = static_cast<int>(std::floor((y + y_half) * inv_dy));
    if (col < 0) col = 0; else if (col >= NCOL) col = NCOL - 1;
    if (row < 0) row = 0; else if (row >= NROW) row = NROW - 1;

    // raw z (meters) into flat buffer by max
    const int lin = row * NCOL + col;
    if (z > out_heights01[lin]) out_heights01[lin] = z;
  }

  // normalize to [0,1] in-place
  for (float& v : out_heights01) {
    float t = (v - zmin) / denom;
    
    v = t;
  }
}

// Write XML to a temp file, then load it with mj_loadXML.
static mjModel* load_model_from_string(const std::string& xml,
                                       char* error, int error_sz) {
  std::string path = "/tmp/mj_hfield_" + std::to_string(::getpid()) + ".xml";
  {
    std::ofstream f(path);
    f << xml;
  }
  mjModel* m = mj_loadXML(path.c_str(), /*vfs*/nullptr, error, error_sz);
  std::remove(path.c_str());  // best-effort cleanup
  return m;
}


// =========================== ROS node =========================== /topic_1/cam_1/depth/color/points
class PC2ToHFieldNode : public rclcpp::Node {
public:
  PC2ToHFieldNode(mjModel* model, mjData* data, int hfield_id,
                  std::vector<float>& heights01, std::mutex& mtx, std::atomic<bool>& flag_new)
  : rclcpp::Node("pc2_to_hfield_minimal_cpp"),
    model_(model), data_(data), hid_(hfield_id),
    heights01_(heights01), mtx_(mtx), new_frame_(flag_new)
  {
    topic_ = this->declare_parameter<std::string>("topic", "/topic_1/cam_1/depth/color/points");
    RCLCPP_INFO(get_logger(), "Subscribing to %s", topic_.c_str());

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      topic_, rclcpp::SensorDataQoS(),
      std::bind(&PC2ToHFieldNode::cbCloud, this, std::placeholders::_1)
    );
  }

private:
  void cbCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Fast field iterators for x,y,z
    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");

    const size_t N = msg->width * msg->height;
    if (N == 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Empty pointcloud");
      return;
    }

    // Copy to plain arrays (tight, contiguous)
    xs_.resize(N);
    ys_.resize(N);
    zs_.resize(N);
    size_t k = 0;
    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
      xs_[k] = *it_x;
      ys_[k] = *it_y;
      zs_[k] = *it_z;
      ++k;
    }
    xs_.resize(k); ys_.resize(k); zs_.resize(k);

    // Bin → normalize, then mark “new frame”
    {
      std::scoped_lock<std::mutex> lock(mtx_);
      project_points_to_grid_max(xs_.data(), ys_.data(), zs_.data(), k, heights01_);
      // we only update the MuJoCo model in the main loop (to avoid heavy work in the callback)
      new_frame_.store(true, std::memory_order_release);
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "PointCloud received: %zu points → grid %dx%d updated", k, NROW, NCOL);
  }

  std::string topic_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;

  // MuJoCo handles (owned elsewhere)
  mjModel* model_;
  mjData*  data_;
  int      hid_;

  // shared grid (normalized 0..1), mutex, and frame flag (owned by main)
  std::vector<float>& heights01_;
  std::mutex&         mtx_;
  std::atomic<bool>&  new_frame_;

  // scratch arrays for incoming cloud
  std::vector<float> xs_, ys_, zs_;
};

// =========================== main ===========================
int main(int argc, char** argv) {
  // ---- MuJoCo: load model from XML string
  const std::string xml = make_model_xml(NROW, NCOL, Lx, Ly, Hz, base);

  char error[1024] = {0};
  mjModel* m = load_model_from_string(xml, error, sizeof(error));
  if (!m) {
    fprintf(stderr, "[MuJoCo] Failed to load model: %s\n", error);
    return 1;
  }
  mjData* d = mj_makeData(m);
  if (!d) {
    fprintf(stderr, "[MuJoCo] Failed to make data.\n");
    mj_deleteModel(m);
    return 1;
  }

  const int hid = mj_name2id(m, mjOBJ_HFIELD, "terrain");
  if (hid < 0) {
    fprintf(stderr, "[MuJoCo] Could not find hfield 'terrain'.\n");
    mj_deleteData(d);
    mj_deleteModel(m);
    return 1;
  }

  // shared heightfield buffer + sync
  std::vector<float> heights01(NROW * NCOL, 0.0f);
  std::mutex heights_mtx;
  std::atomic<bool> new_frame{false};

  // Initialize ROS
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PC2ToHFieldNode>(m, d, hid, heights01, heights_mtx, new_frame);

  // spin ROS in background (like your Python threading)
  std::thread ros_thread([&]() {
    rclcpp::spin(node);
  });
  ros_thread.detach();

  // ---- Simple simulation loop (no viewer, compact)
  // If you want a viewer later, you can set up mjvScene + mjrContext + GLFW.
  printf("[MuJoCo] Running loop. Waiting for heightfield updates...\n");

  // upload the (flat) initial hfield
  set_heightfield(m, hid, heights01);
// main loop
// ---- Viewer setup (GLFW + MuJoCo render context)
if (!glfwInit()) {
    fprintf(stderr, "[GLFW] init failed\n");
    return 1;
}
GLFWwindow* window = glfwCreateWindow(1200, 900, "HField (C++)", NULL, NULL);
if (!window) {
    fprintf(stderr, "[GLFW] create window failed\n");
    glfwTerminate();
    return 1;
}
glfwMakeContextCurrent(window);


glfwSwapInterval(1);

mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;
mjv_defaultCamera(&cam);
cam.lookat[0] = 0.0;   // center X
cam.lookat[1] = 0.0;   // center Y
cam.lookat[2] = 0.5;   // a bit above the plane
cam.azimuth   = 100;    // view angle
cam.elevation = -20;
cam.distance  = 5.0;  // <<< zoomed out (increase if you want more)
mjv_defaultOption(&opt);
mjv_defaultScene(&scn);
mjr_defaultContext(&con);

mjv_makeScene(m, &scn, /*maxgeom*/ 2000);
mjr_makeContext(m, &con, mjFONTSCALE_100);

// upload the initial (flat) hfield to GPU once
set_heightfield(m, hid, heights01);
mjr_uploadHField(m, &con, hid);

printf("[MuJoCo] Viewer running. Close window or Ctrl-C to exit.\n");

// ---- Main render loop
while (!glfwWindowShouldClose(window) && rclcpp::ok()) {
    // If a fresh grid arrived from the ROS callback, write & upload it
    if (new_frame.load(std::memory_order_acquire)) {
        std::scoped_lock<std::mutex> lock(heights_mtx);
        set_heightfield(m, hid, heights01);
        mj_forward(m, d);                 // keep state consistent (optional)
        mjr_uploadHField(m, &con, hid);   // CPU hfield -> GPU
        new_frame.store(false, std::memory_order_release);
    }

    // Step physics a bit (optional)
    mj_step(m, d);

    // Render
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    mjrRect viewport{0, 0, width, height};
    mjv_updateScene(m, d, &opt, /*pert*/nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
    glfwPollEvents();
}

// ---- Clean shutdown (join ROS thread BEFORE destroying contexts)
rclcpp::shutdown();
if (ros_thread.joinable()) ros_thread.join();
mjr_freeContext(&con);
mjv_freeScene(&scn);
glfwDestroyWindow(window);
glfwTerminate();

// Cleanup
rclcpp::shutdown();
  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}
