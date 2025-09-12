// Build with: rclcpp, sensor_msgs, tf2_ros, tf2_sensor_msgs, mujoco, glfw, OpenGL
// One camera, with one transformation, mujoco, glfw - slowish
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>   // tf2::doTransform

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
static constexpr int   NROW = 160;
static constexpr int   NCOL = 160;
static constexpr double Lx   = 6.0;   // meters X span
static constexpr double Ly   = 6.0;   // meters Y span
static constexpr double Hz   = 0.8;   // hfield vertical scale
static constexpr double base = 0.05;  // hfield base offset

static constexpr double ROI_X_HALF = Lx * 0.5;
static constexpr double ROI_Y_HALF = Ly * 0.5;

static constexpr float Z_MIN = -2.0f;  // meters mapped to 0
static constexpr float Z_MAX = 2.0f;  // meters mapped to 1 (tune as needed)

static constexpr double dx = (2.0 * ROI_X_HALF) / double(NCOL);
static constexpr double dy = (2.0 * ROI_Y_HALF) / double(NROW);

// =========================== MuJoCo helpers ===========================
static std::string make_model_xml(int nrow, int ncol, double Lx_, double Ly_, double Hz_, double base_) {
  char buf[4096];
  std::snprintf(buf, sizeof(buf),
R"(<?xml version="1.0"?>
<mujoco model="pc2_onecam_tf">
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

// =========================== Grid projection ===========================
static inline void fuse_points_to_grid_max(
    const float* xs, const float* ys, const float* zs, size_t N,
    std::vector<float>& grid01 /* NROW*NCOL, max-updated */, bool clear_first)
{
  const float xh  = (float)ROI_X_HALF;
  const float yh  = (float)ROI_Y_HALF;
  const float invdx = 1.0f / (float)dx;
  const float invdy = 1.0f / (float)dy;
  const float denom = std::max(1e-6f, Z_MAX - Z_MIN);

  if (clear_first) std::fill(grid01.begin(), grid01.end(), 0.0f);

  for (size_t i = 0; i < N; ++i) {
    float x = xs[i], y = ys[i], z = zs[i];  // already in target frame (base_link)

    if (!std::isfinite(z)) continue;
    if (!(x >= -xh && x < xh && y >= -yh && y < yh)) continue;

    int col = (int)std::floor((x + xh) * invdx);
    int row = (int)std::floor((y + yh) * invdy);
    col = std::clamp(col, 0, NCOL - 1);
    row = std::clamp(row, 0, NROW - 1);
    const int lin = row * NCOL + col;

    // normalize z -> [0,1]
    float t = (z - Z_MIN) / denom;
    if (t > 1.f) t = 1.f; else if (t < 0.f) t = 0.f;
    if (t > grid01[lin]) grid01[lin] = t;
  }
}

// =========================== ROS node (one cam with TF) ===========================
class OneCamTFNode : public rclcpp::Node {
public:
  OneCamTFNode(std::vector<float>& heights01, std::mutex& mtx, std::atomic<bool>& new_frame)
  : rclcpp::Node("pc2_onecam_with_tf"),
    heights01_(heights01), mtx_(mtx), new_frame_(new_frame),
    tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
  {
    topic_        = this->declare_parameter<std::string>("topic", "/topic_1/cam_1/depth/color/points");
    target_frame_ = this->declare_parameter<std::string>("target_frame", "base_link");

    RCLCPP_INFO(get_logger(), "Subscribing to %s, transforming into %s", topic_.c_str(), target_frame_.c_str());

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      topic_, rclcpp::SensorDataQoS(),
      std::bind(&OneCamTFNode::cbCloud, this, std::placeholders::_1)
    );
  }

private:
  void cbCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    const size_t N = (size_t)msg->width * msg->height;
    if (N == 0) return;

    // 1) TF: source -> target_frame_ at msg time
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(
        target_frame_, msg->header.frame_id, msg->header.stamp,
        std::chrono::milliseconds(50));
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "[TF] %s <- %s failed: %s",
        target_frame_.c_str(), msg->header.frame_id.c_str(), e.what());
      return;
    }

    // 2) Transform cloud into target_frame_
    sensor_msgs::msg::PointCloud2 cloud_tf;
    try {
      tf2::doTransform(*msg, cloud_tf, tf);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "[TF] doTransform failed: %s", e.what());
      return;
    }

    // 3) Read XYZ and fuse into grid01
    try {
      sensor_msgs::PointCloud2ConstIterator<float> itx(cloud_tf, "x");
      sensor_msgs::PointCloud2ConstIterator<float> ity(cloud_tf, "y");
      sensor_msgs::PointCloud2ConstIterator<float> itz(cloud_tf, "z");

      const size_t Nt = (size_t)cloud_tf.width * cloud_tf.height;
      std::vector<float> xs; xs.resize(Nt);
      std::vector<float> ys; ys.resize(Nt);
      std::vector<float> zs; zs.resize(Nt);

      size_t k = 0;
      for (; itx != itx.end(); ++itx, ++ity, ++itz) {
        xs[k] = *itx; ys[k] = *ity; zs[k] = *itz; ++k;
      }
      xs.resize(k); ys.resize(k); zs.resize(k);

      {
        std::scoped_lock<std::mutex> lk(mtx_);
        // clear_first = true since it's single-cam (each msg can define a frame)
        fuse_points_to_grid_max(xs.data(), ys.data(), zs.data(), k, heights01_, /*clear_first=*/true);
        new_frame_.store(true, std::memory_order_release);
      }

      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "[%s] %zu pts fused into %dx%d", target_frame_.c_str(), k, NROW, NCOL);

    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "[Iterator] error: %s", e.what());
    }
  }

  // ROS
  std::string topic_, target_frame_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Shared grid
  std::vector<float>& heights01_;
  std::mutex&         mtx_;
  std::atomic<bool>&  new_frame_;
};

// =========================== main ===========================
int main(int argc, char** argv) {
  // MuJoCo model
  const std::string xml = make_model_xml(NROW, NCOL, Lx, Ly, Hz, base);
  char error[1024] = {0};
  mjModel* m = load_model_from_string(xml, error, sizeof(error));
  if (!m) { std::fprintf(stderr, "mj_loadXML error: %s\n", error); return 1; }
  mjData* d = mj_makeData(m);
  const int hid = mj_name2id(m, mjOBJ_HFIELD, "terrain");
  if (hid < 0) { std::fprintf(stderr, "hfield 'terrain' not found\n"); return 1; }

  // Shared heightfield buffer + sync
  std::vector<float> heights01(NROW*NCOL, 0.0f);
  std::mutex heights_mtx;
  std::atomic<bool> new_frame{false};

  // ROS
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OneCamTFNode>(heights01, heights_mtx, new_frame);
  std::thread ros_thread([&]{ rclcpp::spin(node); });

  // Viewer (GLFW + MuJoCo render thread)
  if (!glfwInit()) { std::fprintf(stderr, "GLFW init failed\n"); return 1; }
  GLFWwindow* win = glfwCreateWindow(1200, 900, "HField (1 cam, with TF)", nullptr, nullptr);
  if (!win) { std::fprintf(stderr, "GLFW window failed\n"); return 1; }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  mjvCamera cam; mjv_defaultCamera(&cam);
  cam.type = mjCAMERA_FREE; cam.distance = 10.5; cam.azimuth = 100; cam.elevation = -25; cam.lookat[2] = 0.4;
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
      // Clear so each new msg paints a fresh frame
      std::fill(heights01.begin(), heights01.end(), 0.0f);
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
