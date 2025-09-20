// one camera one transform, it does the math for one time extraction of the TF and puts in it in a hardcoded R,t
// Python equivalent: one_cam_one_tf.py
// pretty fast, however if I I always use same static TF, I can hardcode R,t in C++ and skip TF lookup
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstring>

static constexpr const char* TOPIC        = "/topic_1/cam_1/depth/color/points";
static constexpr const char* TARGET_FRAME = "base_link";

// Grid & world
static constexpr int    NROW = 160;
static constexpr int    NCOL = 160;
static constexpr double Lx   = 6.0;
static constexpr double Ly   = 6.0;
static constexpr double Hz   = 0.8;
static constexpr double base = 0.05;

static constexpr double ROI_X_HALF = Lx * 0.5;
static constexpr double ROI_Y_HALF = Ly * 0.5;

// z → [0,1] mapping (meters in TARGET_FRAME)
static constexpr float Z_MIN = -1.0f;
static constexpr float Z_MAX =  1.0f;

static constexpr double dx = (2.0 * ROI_X_HALF) / double(NCOL);
static constexpr double dy = (2.0 * ROI_Y_HALF) / double(NROW);

// ---------- MuJoCo helpers ----------
static std::string make_model_xml(int nrow=NROW, int ncol=NCOL,
                                  double Lx_=Lx, double Ly_=Ly, double Hz_=Hz, double base_=base)
{
  char buf[4096];
  std::snprintf(buf, sizeof(buf),
R"(<?xml version="1.0"?>
<mujoco model="pc2_onecam_tf_fast">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <asset>
    <hfield name="terrain" nrow="%d" ncol="%d" size="%.3f %.3f %.3f %.3f"/>
  </asset>
  <worldbody>
    <light name="toplight" pos="0 0 5" dir="0 0 -1" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3" directional="true"/>
    <geom type="hfield" hfield="terrain" rgba="0.7 0.7 0.7 1"/>
    <camera name="iso" pos="2.0 -2.5 1.8" euler="35 0 25"/>
  </worldbody>
</mujoco>
)", nrow, ncol, Lx_, Ly_, Hz_, base_);
  return std::string(buf);
}

static mjModel* load_model_from_string(const std::string& xml, char* error, int error_sz) {
  std::string path = "/tmp/mj_hfield_onecam.xml";
  {
    std::ofstream f(path);
    f << xml;
  }
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

// ---------- Math helpers ----------
static inline void quat_to_rot(float qx, float qy, float qz, float qw, float R[9]) {
  double n = std::sqrt(double(qx)*qx + double(qy)*qy + double(qz)*qz + double(qw)*qw);
  if (n <= 0.0) {
    R[0]=1; R[1]=0; R[2]=0;
    R[3]=0; R[4]=1; R[5]=0;
    R[6]=0; R[7]=0; R[8]=1;
    return;
  }
  double x = qx/n, y = qy/n, z = qz/n, w = qw/n;
  double xx=x*x, yy=y*y, zz=z*z;
  double xy=x*y, xz=x*z, yz=y*z;
  double wx=w*x, wy=w*y, wz=w*z;

  R[0] = float(1 - 2*(yy+zz)); R[1] = float(2*(xy - wz));   R[2] = float(2*(xz + wy));
  R[3] = float(2*(xy + wz));   R[4] = float(1 - 2*(xx+zz)); R[5] = float(2*(yz - wx));
  R[6] = float(2*(xz - wy));   R[7] = float(2*(yz + wx));   R[8] = float(1 - 2*(xx+yy));
}

// ---------- ROS2 node ----------
class OneCamTFFastNode : public rclcpp::Node {
public:
  OneCamTFFastNode(std::vector<float>& heights01, std::mutex& mtx, std::atomic<bool>& new_frame)
  : rclcpp::Node("pc2_onecam_tf_fast_cpp"),
    heights01_(heights01), mtx_(mtx), new_frame_(new_frame),
    tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
  {
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      TOPIC, rclcpp::SensorDataQoS(),
      std::bind(&OneCamTFFastNode::cbCloud, this, std::placeholders::_1)
    );
    RCLCPP_INFO(get_logger(), "Subscribing to %s; transforming into %s", TOPIC, TARGET_FRAME);
  }

private:
  void cbCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // --- 1) TF lookup (source -> TARGET_FRAME) ---
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(
        TARGET_FRAME, msg->header.frame_id,
        rclcpp::Time(msg->header.stamp),
        std::chrono::milliseconds(50));
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "[TF] %s <- %s failed: %s", TARGET_FRAME, msg->header.frame_id.c_str(), e.what());
      return;
    }

    const auto& t = tf.transform.translation;
    const auto& q = tf.transform.rotation;

    float Rm[9];
    quat_to_rot((float)q.x, (float)q.y, (float)q.z, (float)q.w, Rm);
    const float tx = (float)t.x, ty = (float)t.y, tz = (float)t.z;

    // --- 2) Local normalized grid (0..1), max fuse ---
    std::vector<float> local(NROW*NCOL, 0.0f);
    const float xh = (float)ROI_X_HALF, yh = (float)ROI_Y_HALF;
    const float invdx = 1.0f / (float)dx, invdy = 1.0f / (float)dy;
    const float denom = std::max(1e-6f, (Z_MAX - Z_MIN));

    sensor_msgs::PointCloud2ConstIterator<float> itx(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> ity(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> itz(*msg, "z");

    size_t fused = 0;
    for (; itx != itx.end(); ++itx, ++ity, ++itz) {
      float xc = *itx, yc = *ity, zc = *itz;
      if (!std::isfinite(xc) || !std::isfinite(yc) || !std::isfinite(zc)) continue;

      // Transform point: p_base = R * p_cam + t
      // Rm is row-major 3x3
      float xb = Rm[0]*xc + Rm[1]*yc + Rm[2]*zc + tx;
      float yb = Rm[3]*xc + Rm[4]*yc + Rm[5]*zc + ty;
      float zb = Rm[6]*xc + Rm[7]*yc + Rm[8]*zc + tz;

      if (!(xb >= -xh && xb < xh && yb >= -yh && yb < yh)) continue;
      if (!std::isfinite(zb)) continue;

      int col = (int)std::floor((xb + xh) * invdx);
      int row = (int)std::floor((yb + yh) * invdy);
      if (col < 0) col = 0; else if (col >= NCOL) col = NCOL - 1;
      if (row < 0) row = 0; else if (row >= NROW) row = NROW - 1;

      float t01 = (zb - Z_MIN) / denom;
      if (t01 < 0.f) t01 = 0.f; else if (t01 > 1.f) t01 = 1.f;

      float& cell = local[row*NCOL + col];
      if (t01 > cell) cell = t01;
      ++fused;
    }

    // --- 3) Publish to shared buffer for viewer thread ---
    {
      std::scoped_lock<std::mutex> lk(mtx_);
      heights01_ = std::move(local);   // copy/move normalized grid
      new_frame_.store(true, std::memory_order_release);
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "[%s] fused %zu pts into %dx%d", TARGET_FRAME, fused, NROW, NCOL);
  }

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Shared grid
  std::vector<float>& heights01_;
  std::mutex&         mtx_;
  std::atomic<bool>&  new_frame_;
};

// ---------- main ----------
int main(int argc, char** argv) {
  // MuJoCo
  char error[1024] = {0};
  std::string xml = make_model_xml();
  mjModel* m = load_model_from_string(xml, error, sizeof(error));
  if (!m) { std::fprintf(stderr, "mj_loadXML error: %s\n", error); return 1; }
  mjData* d = mj_makeData(m);
  int hid = mj_name2id(m, mjOBJ_HFIELD, "terrain");
  if (hid < 0) { std::fprintf(stderr, "hfield 'terrain' not found\n"); return 1; }

  // Shared normalized grid + sync
  std::vector<float> heights01(NROW*NCOL, 0.0f);
  std::mutex heights_mtx;
  std::atomic<bool> new_frame{false};

  // ROS
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OneCamTFFastNode>(heights01, heights_mtx, new_frame);
  std::thread ros_thread([&](){ rclcpp::spin(node); });
  ros_thread.detach();

  // Viewer
  if (!glfwInit()) { std::fprintf(stderr, "GLFW init failed\n"); return 1; }
  GLFWwindow* win = glfwCreateWindow(1200, 900, "HField (1 cam, TF vector)", nullptr, nullptr);
  if (!win) { std::fprintf(stderr, "GLFW window failed\n"); glfwTerminate(); return 1; }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  mjvCamera cam; mjv_defaultCamera(&cam);
  cam.type = mjCAMERA_FREE; cam.distance = 8.0f; cam.azimuth = 100; cam.elevation = -25; cam.lookat[2] = 0.4f;
  mjvOption opt; mjv_defaultOption(&opt);
  mjvScene scn; mjv_defaultScene(&scn);
  mjrContext con; mjr_defaultContext(&con);
  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_100);

  // Initial upload
  set_heightfield_cpu(m, hid, heights01);
  upload_heightfield_gpu(m, hid, &con);

  // Main loop (render thread owns GL)
  while (!glfwWindowShouldClose(win) && rclcpp::ok()) {
    bool do_upload = false;
    {
      if (new_frame.load(std::memory_order_acquire)) {
        std::scoped_lock<std::mutex> lk(heights_mtx);
        set_heightfield_cpu(m, hid, heights01);
        do_upload = true;
        new_frame.store(false, std::memory_order_release);
      }
    }
    if (do_upload) upload_heightfield_gpu(m, hid, &con);

    mj_step(m, d);
    int W,H; glfwGetFramebufferSize(win, &W, &H);
    mjrRect vp{0,0,W,H};
    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(vp, &scn, &con);
    glfwSwapBuffers(win);
    glfwPollEvents();
  }

  rclcpp::shutdown();
  mjr_freeContext(&con); mjv_freeScene(&scn);
  glfwDestroyWindow(win); glfwTerminate();
  mj_deleteData(d); mj_deleteModel(m);
  return 0;
}
