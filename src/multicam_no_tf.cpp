// up to 3 cameras, no transformations, mujoco, glfw - fast
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <limits>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <unistd.h>

// =========================== CONFIG ===========================
static const std::vector<std::string> CAM_TOPICS = {
    "/topic_1/cam_1/depth/color/points",
    "/topic_2/cam_2/depth/color/points",
    "/topic_3/cam_3/depth/color/points",
};

static constexpr int   NROW = 160;
static constexpr int   NCOL = 160;
static constexpr double Lx   = 6.0;
static constexpr double Ly   = 6.0;
static constexpr double Hz   = 0.8;
static constexpr double base = 0.05;

static constexpr double ROI_X_HALF = Lx * 0.5;
static constexpr double ROI_Y_HALF = Ly * 0.5;

static constexpr float Z_MIN = 0.0f;
static constexpr float Z_MAX = 1.0f;

static constexpr double dx = (2.0 * ROI_X_HALF) / double(NCOL);
static constexpr double dy = (2.0 * ROI_Y_HALF) / double(NROW);

// =========================== MuJoCo helpers ===========================
static std::string make_model_xml(int nrow, int ncol, double Lx_, double Ly_, double Hz_, double base_) {
  char buf[4096];
  std::snprintf(buf, sizeof(buf),
R"(<?xml version="1.0"?>
<mujoco model="pc2_live_hfield">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <asset>
    <hfield name="terrain" nrow="%d" ncol="%d" size="%.3f %.3f %.3f %.3f"/>
  </asset>
  <worldbody>
    <light name="toplight" pos="0 0 5" dir="0 0 -1" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3" directional="true"/>
    <geom type="hfield" hfield="terrain" rgba="0.7 0.7 0.7 1"/>
    <camera name="iso" pos="2.0 -2.5 1.8" euler="35 0 25" fovy="60"/>
  </worldbody>
</mujoco>
)", nrow, ncol, Lx_, Ly_, Hz_, base_);
  return std::string(buf);
}

static mjModel* load_model_from_string(const std::string& xml, char* error, int error_sz) {
  std::string path = "/tmp/mj_hfield_" + std::to_string(::getpid()) + ".xml";
  { std::ofstream f(path); f << xml; }
  mjModel* m = mj_loadXML(path.c_str(), nullptr, error, error_sz);
  std::remove(path.c_str());
  return m;
}

static inline void set_heightfield_cpu(mjModel* m, int hid, const std::vector<float>& heights01) {
  if (!m || hid < 0 || hid >= m->nhfield) return;
  const int nrow = m->hfield_nrow[hid];
  const int ncol = m->hfield_ncol[hid];
  if ((int)heights01.size() != nrow * ncol) return;
  const int adr = m->hfield_adr[hid];
  std::memcpy(m->hfield_data + adr, heights01.data(), sizeof(float)*nrow*ncol);
}

static inline void upload_heightfield_gpu(mjModel* m, int hid, const mjrContext* con) {
  if (!m || !con || hid < 0 || hid >= m->nhfield) return;
  mjr_uploadHField(m, con, hid);
}

// =========================== Grid projection (no TF) ===========================
static inline void fuse_points_to_grid_max_no_tf(
    const float* xs, const float* ys, const float* zs, size_t N,
    std::vector<float>& grid01 /* NROW*NCOL, max-updated in place */)
{
  const float xh  = (float)ROI_X_HALF;
  const float yh  = (float)ROI_Y_HALF;
  const float invdx = 1.0f / (float)dx;
  const float invdy = 1.0f / (float)dy;
  const float denom = std::max(1e-6f, Z_MAX - Z_MIN);

  for (size_t i = 0; i < N; ++i) {
    float x = xs[i], y = ys[i], z = zs[i];

    // Example z transform (match earlier tests); tweak/remove as you like:
    z = -z + 1.0f;
    if (!std::isfinite(z)) continue;

    if (!(x >= -xh && x < xh && y >= -yh && y < yh)) continue;

    int col = (int)std::floor((x + xh) * invdx);
    int row = (int)std::floor((y + yh) * invdy);
    col = std::clamp(col, 0, NCOL - 1);
    row = std::clamp(row, 0, NROW - 1);
    const int lin = row * NCOL + col;

    // normalize to [0,1] on the fly
    float t = (z - Z_MIN) / denom;
    if (t > grid01[lin]) grid01[lin] = t;
  }
}

// =========================== Multi-cam ROS node (no TF) ===========================
class MultiPC2NoTFNode : public rclcpp::Node {
public:
  MultiPC2NoTFNode(std::vector<float>& heights01, std::mutex& mtx, std::atomic<bool>& new_frame)
  : rclcpp::Node("pc2_no_tf_multicam"), heights01_(heights01), mtx_(mtx), new_frame_(new_frame)
  {
    auto qos = rclcpp::SensorDataQoS();
    subs_.reserve(CAM_TOPICS.size());
    xs_.resize(CAM_TOPICS.size());
    ys_.resize(CAM_TOPICS.size());
    zs_.resize(CAM_TOPICS.size());

    for (size_t i = 0; i < CAM_TOPICS.size(); ++i) {
      const auto& topic = CAM_TOPICS[i];
      RCLCPP_INFO(get_logger(), "Subscribing to %s", topic.c_str());
      subs_.push_back(
        create_subscription<sensor_msgs::msg::PointCloud2>(
          topic, qos, [this, i, topic](const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            this->cbCloud(i, topic, msg);
          })
      );
    }
  }

private:
  void cbCloud(size_t idx, const std::string& topic, const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const size_t N = (size_t)msg->width * msg->height;
    if (N == 0) return;

    sensor_msgs::PointCloud2ConstIterator<float> itx(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> ity(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> itz(*msg, "z");

    auto& xs = xs_[idx]; auto& ys = ys_[idx]; auto& zs = zs_[idx];
    xs.resize(N); ys.resize(N); zs.resize(N);

    size_t k = 0;
    for (; itx != itx.end(); ++itx, ++ity, ++itz) {
      xs[k] = *itx; ys[k] = *ity; zs[k] = *itz; ++k;
    }
    xs.resize(k); ys.resize(k); zs.resize(k);

    // Fuse into shared grid (max). We DO NOT clear the grid here; main loop clears each frame.
    {
      std::scoped_lock<std::mutex> lk(mtx_);
      fuse_points_to_grid_max_no_tf(xs.data(), ys.data(), zs.data(), k, heights01_);
      new_frame_.store(true, std::memory_order_release);
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "[%s] %zu pts fused", topic.c_str(), k);
  }

  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subs_;
  std::vector<std::vector<float>> xs_, ys_, zs_;

  std::vector<float>& heights01_;
  std::mutex&         mtx_;
  std::atomic<bool>&  new_frame_;
};

// =========================== main ===========================
int main(int argc, char** argv) {
  // MuJoCo: model
  const std::string xml = make_model_xml(NROW, NCOL, Lx, Ly, Hz, base);
  char error[1024] = {0};
  mjModel* m = load_model_from_string(xml, error, sizeof(error));
  if (!m) { std::fprintf(stderr, "mj_loadXML error: %s\n", error); return 1; }
  mjData* d = mj_makeData(m);
  const int hid = mj_name2id(m, mjOBJ_HFIELD, "terrain");
  if (hid < 0) { std::fprintf(stderr, "hfield 'terrain' not found\n"); return 1; }

  // Shared heightfield (normalized 0..1)
  std::vector<float> heights01(NROW*NCOL, 0.0f);
  std::mutex heights_mtx;
  std::atomic<bool> new_frame{false};

  // ROS
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MultiPC2NoTFNode>(heights01, heights_mtx, new_frame);
  std::thread ros_thread([&]{ rclcpp::spin(node); });

  // Viewer (GLFW)
  if (!glfwInit()) { std::fprintf(stderr, "GLFW init failed\n"); return 1; }
  GLFWwindow* win = glfwCreateWindow(1200, 900, "HField (3 cams, no TF)", nullptr, nullptr);
  if (!win) { std::fprintf(stderr, "GLFW window failed\n"); return 1; }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  mjvCamera cam; mjv_defaultCamera(&cam);
  cam.type = mjCAMERA_FREE; cam.distance = 6.0; cam.azimuth = 45; cam.elevation = -25; cam.lookat[2] = 0.4;
  mjvOption opt; mjv_defaultOption(&opt);
  mjvScene scn; mjv_defaultScene(&scn);
  mjrContext con; mjr_defaultContext(&con);
  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_100);

  // Initial upload
  set_heightfield_cpu(m, hid, heights01);
  upload_heightfield_gpu(m, hid, &con);

  // Main loop
  while (!glfwWindowShouldClose(win) && rclcpp::ok()) {
    bool do_upload = false;

    if (new_frame.load(std::memory_order_acquire)) {
      std::scoped_lock<std::mutex> lk(heights_mtx);
      set_heightfield_cpu(m, hid, heights01);
      do_upload = true;

      // Clear for next frame’s fusion
      std::fill(heights01.begin(), heights01.end(), Z_MIN);
      new_frame.store(false, std::memory_order_release);
    }

    if (do_upload) upload_heightfield_gpu(m, hid, &con);

    mj_step(m, d);
    int W, H; glfwGetFramebufferSize(win, &W, &H);
    mjrRect vp{0,0,W,H};
    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(vp, &scn, &con);

    glfwSwapBuffers(win);
    glfwPollEvents();
  }

  // Shutdown
  rclcpp::shutdown();
  if (ros_thread.joinable()) ros_thread.join();
  mjr_freeContext(&con); mjv_freeScene(&scn);
  glfwDestroyWindow(win); glfwTerminate();
  mj_deleteData(d); mj_deleteModel(m);
  return 0;
}
