// g++/CMake with: rclcpp sensor_msgs tf2_ros MuJoCo GLFW
// Two-camera → TF to base_link → MuJoCo heightfield (vectorized transform, max-fuse)
// translation from two_cam_two_tf.py. added a third camera and it was fine

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

using std::placeholders::_1;

// =========================== CONFIG ===========================
static const char* CAM_TOPICS[3] = {
  "/topic_1/cam_1/depth/color/points",
  "/topic_2/cam_2/depth/color/points",
  "/topic_3/cam_3/depth/color/points"
};
static const char* TARGET_FRAME = "base_link";

constexpr int   NROW = 160;
constexpr int   NCOL = 160;
constexpr float Lx   = 6.0f;
constexpr float Ly   = 6.0f;
constexpr float Hz   = 0.8f;   // vertical exaggeration in MuJoCo
constexpr float base = 0.05f;  // base offset in MuJoCo

constexpr float ROI_X_HALF = Lx * 0.5f;
constexpr float ROI_Y_HALF = Ly * 0.5f;

constexpr float Z_MIN = -1.0f; // meters
constexpr float Z_MAX =  1.0f; // meters

constexpr float dx = (2.0f * ROI_X_HALF) / NCOL;
constexpr float dy = (2.0f * ROI_Y_HALF) / NROW;

// =========================== MuJoCo helpers ===========================
static std::string make_model_xml(int nrow=NROW, int ncol=NCOL,
                                  float Lx=Lx, float Ly=Ly, float Hz=Hz, float base=base) {
  char buf[2048];
  std::snprintf(buf, sizeof(buf),
  // don't keep updating the model. use adrians code
R"XML(<mujoco model='pc2_live_hfield'>
  <option timestep='0.005' gravity='0 0 -9.81'/>
  <asset>
    <hfield name='terrain' nrow='160' ncol='160' size='6.0 6.0 0.8 0.05'/>
  </asset>
  <worldbody>
    <light name="toplight" pos="0 0 5" dir="0 0 -1" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3" directional="true"/>
    <geom type='hfield' hfield='terrain' rgba='0.7 0.7 0.7 1'/>
    <camera name='iso' pos='2.0 -2.5 1.8' euler='35 0 25'/>
  </worldbody>
</mujoco>)XML",
    nrow, ncol, Lx, Ly, Hz, base);
  return std::string(buf);
}

static inline void set_heightfield(mjModel* m, int hid, const std::vector<float>& arr) {
  const int nrow = m->hfield_nrow[hid];
  const int ncol = m->hfield_ncol[hid];
  const size_t n = static_cast<size_t>(nrow) * static_cast<size_t>(ncol);
  const int adr = m->hfield_adr[hid];
  float* dest = &m->hfield_data[adr];
  std::copy(arr.begin(), arr.begin() + n, dest);
}



static inline void upload_heightfield(mjModel* m, int hid, mjrContext* con) {
  mjr_uploadHField(m, con, hid);
}

// =========================== Math helpers ===========================
struct Mat3 {
  float a11,a12,a13, a21,a22,a23, a31,a32,a33;
};
static Mat3 quat_to_rot(float qx, float qy, float qz, float qw) {
  // normalize
  float n = std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
  if (n == 0.0f) {
    return {1,0,0, 0,1,0, 0,0,1};
  }
  qx/=n; qy/=n; qz/=n; qw/=n;
  float xx=qx*qx, yy=qy*qy, zz=qz*qz;
  float xy=qx*qy, xz=qx*qz, yz=qy*qz;
  float wx=qw*qx, wy=qw*qy, wz=qw*qz;
  Mat3 R;
  R.a11 = 1 - 2*(yy+zz); R.a12 = 2*(xy - wz);   R.a13 = 2*(xz + wy);
  R.a21 = 2*(xy + wz);   R.a22 = 1 - 2*(xx+zz); R.a23 = 2*(yz - wx);
  R.a31 = 2*(xz - wy);   R.a32 = 2*(yz + wx);   R.a33 = 1 - 2*(xx+yy);
  return R;
}

// =========================== Fusion ===========================
static inline void fuse_points_to_grid_meters(const std::vector<float>& pts_xyz, // flat x,y,z...
                                              std::vector<float>& grid_m) {
  // pts_xyz size is 3*N; grid_m is NROW*NCOL (row-major on y,x)
  const size_t N = pts_xyz.size() / 3;
  float* gm = grid_m.data();

  for (size_t i=0; i<N; ++i) {
    float x = pts_xyz[3*i+0];
    float y = pts_xyz[3*i+1];
    float z = pts_xyz[3*i+2];

    if (!(x >= -ROI_X_HALF && x < ROI_X_HALF &&
          y >= -ROI_Y_HALF && y < ROI_Y_HALF) ||
        !std::isfinite(z)) {
      continue;
    }
    int col = static_cast<int>((x + ROI_X_HALF) / dx);
    int row = static_cast<int>((y + ROI_Y_HALF) / dy);
    if (col < 0) {col = 0;} 
    else if (col >= NCOL) col = NCOL-1;
    if (row < 0) {row = 0;} 
    else if (row >= NROW) row = NROW-1;

    size_t lin = static_cast<size_t>(row) * NCOL + static_cast<size_t>(col);
    gm[lin] = std::max(gm[lin], z);
  }
}

// =========================== Node ===========================
class MultiCamNode : public rclcpp::Node {
public:
  MultiCamNode(mjModel* m, mjData* d, int hid, mjrContext* con)
  : Node("multicam_pc2_to_hfield"),
    model_(m), data_(d), hid_(hid), con_(con),
    tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {

    heights01_.assign(NROW*NCOL, 0.0f);
    grid_m_.assign(NROW*NCOL, -std::numeric_limits<float>::infinity());

    rclcpp::SensorDataQoS qos;

    for (const char* topic : CAM_TOPICS) {
      subs_.push_back(this->create_subscription<sensor_msgs::msg::PointCloud2>(
        topic, qos,
        [this, topic](const sensor_msgs::msg::PointCloud2::SharedPtr msg){
          this->callback(msg, topic);
        }));
      RCLCPP_INFO(this->get_logger(), "Subscribed to %s", topic);
    }

    // periodic on_tick @ 50 Hz
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(16), // do 16 for 62.5 hz, use 25 for 30hz
      std::bind(&MultiCamNode::on_tick, this));

    RCLCPP_INFO(this->get_logger(), "Fusing 2 cameras into %s. ROI=(%.2fm x %.2fm), grid=%dx%d",
                TARGET_FRAME, Lx, Ly, NROW, NCOL);
  }

  void on_tick() {
    if (!new_frame_.exchange(false)) return;

    // Copy meters grid
    std::vector<float> grid_copy;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      grid_copy = grid_m_;
    }

    // Replace -inf with Z_MIN
    for (float& v : grid_copy) {
      if (!std::isfinite(v)) v = Z_MIN;
    }

    // // Normalize to [0,1]
    // const float denom = std::max(1e-6f, Z_MAX - Z_MIN);
    // for (float& v : grid_copy) {
    //   v = (v - Z_MIN) / denom;
    //   // if (v < 0.0f) v = 0.0f; else if (v > 1.0f) v = 1.0f;
    // }

    // Push to MuJoCo and reset raw meters grid
    {
      std::lock_guard<std::mutex> lk(mtx_);
      heights01_ = grid_copy;
      heights01_stage_ = heights01_; 
      // set_heightfield(model_, hid_, heights01_);
      // upload_heightfield(model_, hid_, con_);
      std::fill(grid_m_.begin(), grid_m_.end(),
                -std::numeric_limits<float>::infinity());
      hfield_dirty_.store(true); 
    }
  }
  bool pop_staged(std::vector<float>& out) {
    if (!hfield_dirty_.exchange(false)) return false;
    std::lock_guard<std::mutex> lk(mtx_);
    out = heights01_stage_;        // copy out
    return true;
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr& msg, const char* topic) {
    // 1) TF lookup (one per message)
    geometry_msgs::msg::TransformStamped tf;
    try {
      // latest available (for strict stamping, use msg->header.stamp)
      tf = tf_buffer_.lookupTransform(
        TARGET_FRAME, msg->header.frame_id, tf2::TimePointZero,
        std::chrono::milliseconds(50));
    } catch (const std::exception& e) {
      RCLCPP_WARN(this->get_logger(), "[%s] TF %s <- %s failed: %s",
                  topic, TARGET_FRAME, msg->header.frame_id.c_str(), e.what());
      return;
    }

    // compute R, t
    const auto& t = tf.transform.translation;
    const auto& q = tf.transform.rotation;
    Mat3 R = quat_to_rot(static_cast<float>(q.x), static_cast<float>(q.y),
                         static_cast<float>(q.z), static_cast<float>(q.w));
    const float tx = static_cast<float>(t.x);
    const float ty = static_cast<float>(t.y);
    const float tz = static_cast<float>(t.z);

    // 2) Extract xyz -> vector<float> (x,y,z,...)
    std::vector<float> pts_xyz;
    pts_xyz.reserve(static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height) * 3);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      const float x = *iter_x;
      const float y = *iter_y;
      const float z = *iter_z;
      if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
        pts_xyz.push_back(x);
        pts_xyz.push_back(y);
        pts_xyz.push_back(z);
      }
    }
    if (pts_xyz.empty()) return;

    // 3) Vectorized-ish transform: x' = R x + t (apply per point)
    for (size_t i=0; i<pts_xyz.size(); i+=3) {
      float x = pts_xyz[i+0], y = pts_xyz[i+1], z = pts_xyz[i+2];
      float X = R.a11*x + R.a12*y + R.a13*z + tx;
      float Y = R.a21*x + R.a22*y + R.a23*z + ty;
      float Z = R.a31*x + R.a32*y + R.a33*z + tz;
      pts_xyz[i+0] = X; pts_xyz[i+1] = Y; pts_xyz[i+2] = Z;
    }

    // 4) Fuse into shared meters grid
    {
      std::lock_guard<std::mutex> lk(mtx_);
      fuse_points_to_grid_meters(pts_xyz, grid_m_);
      new_frame_.store(true);
    }
  }

  // MuJoCo
  mjModel* model_;
  mjData*  data_;
  int      hid_;
  mjrContext* con_;

  // Shared buffers
  std::vector<float> heights01_;       // normalized [0,1]
  std::vector<float> grid_m_;          // meters, max-fused, -inf when empty
  std::mutex mtx_;
  std::atomic<bool> new_frame_{false};
  std::vector<float> heights01_stage_;
  std::atomic<bool> hfield_dirty_{false};

  
  // ROS
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subs_;
  rclcpp::TimerBase::SharedPtr timer_;
};

// =========================== Minimal MuJoCo viewer bootstrap ===========================
static GLFWwindow* make_window(int w=1200, int h=900) {
  if (!glfwInit()) return nullptr;
  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  GLFWwindow* win = glfwCreateWindow(w, h, "HField Live", nullptr, nullptr);
  if (!win) { glfwTerminate(); return nullptr; }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);
  return win;
}

// ----start of mouse
// --- Mouse-controlled camera helpers ---
struct MouseState {
  bool left=false, middle=false, right=false;
  double lastx=0, lasty=0;
};

struct AppCtx {
  mjModel* m;
  mjvScene* scn;
  mjvCamera* cam;
  MouseState mouse;
};

// Mouse buttons: track which is down and remember last cursor position
static void mouse_button_cb(GLFWwindow* w, int button, int action, int mods) {
  auto* a = static_cast<AppCtx*>(glfwGetWindowUserPointer(w));
  if (!a) return;
  if (button == GLFW_MOUSE_BUTTON_LEFT)   a->mouse.left   = (action == GLFW_PRESS);
  if (button == GLFW_MOUSE_BUTTON_MIDDLE) a->mouse.middle = (action == GLFW_PRESS);
  if (button == GLFW_MOUSE_BUTTON_RIGHT)  a->mouse.right  = (action == GLFW_PRESS);
  glfwGetCursorPos(w, &a->mouse.lastx, &a->mouse.lasty);
}

// Cursor movement: rotate (RMB), pan (MMB)
static void cursor_pos_cb(GLFWwindow* w, double xpos, double ypos) {
  auto* a = static_cast<AppCtx*>(glfwGetWindowUserPointer(w));
  if (!a) return;

  double dx = xpos - a->mouse.lastx;
  double dy = ypos - a->mouse.lasty;
  a->mouse.lastx = xpos;
  a->mouse.lasty = ypos;

  // scale a bit for nicer feel
  const double rot_scale  = 0.005;
  const double move_scale = 0.05;

  if (a->mouse.right) {
    // rotate: horizontal then vertical
    mjv_moveCamera(a->m, mjMOUSE_ROTATE_H,  dx*rot_scale, 0,            a->scn, a->cam);
    mjv_moveCamera(a->m, mjMOUSE_ROTATE_V,  0,            -dy*rot_scale, a->scn, a->cam);
  } else if (a->mouse.middle) {
    // pan: horizontal then vertical
    mjv_moveCamera(a->m, mjMOUSE_MOVE_H,    dx*move_scale, 0,            a->scn, a->cam);
    mjv_moveCamera(a->m, mjMOUSE_MOVE_V,    0,             -dy*move_scale, a->scn, a->cam);
  }
}

// Scroll wheel: zoom
static void scroll_cb(GLFWwindow* w, double xoff, double yoff) {
  auto* a = static_cast<AppCtx*>(glfwGetWindowUserPointer(w));
  if (!a) return;
  const double zoom_scale = 0.05;
  mjv_moveCamera(a->m, mjMOUSE_ZOOM, 0, -yoff*zoom_scale, a->scn, a->cam);
}
// -----end of mouse

int main(int argc, char** argv) {
  // ---- MuJoCo
  std::string xml = make_model_xml();
  std::string tmp_path = "/tmp/twocam_hfield.xml";

  {
    std::ofstream f(tmp_path);
    f << xml;
  }
  char error[1024] = {0};
  mjModel* m = mj_loadXML(tmp_path.c_str(), /*vfs*/nullptr, error, sizeof(error));
  if (!m) {
    std::fprintf(stderr, "MuJoCo load error: %s\n", error);
    return 1;
  }
  mjData* d = mj_makeData(m);

  GLFWwindow* win = make_window();
  if (!win) { mj_deleteData(d); mj_deleteModel(m); return 1; }

  // Visualization helpers
  mjvCamera cam;      mjv_defaultCamera(&cam);
  cam.type = mjCAMERA_FREE;
  
  // cam.distance = 15;
  
    
  //   cam.lookat[0] = 0;  cam.lookat[1] = 0;  cam.lookat[2] = 0.5; // target
  //    // bigger = farther = zoomed out
  //   cam.azimuth    = 100;    // deg around z
  //   cam.elevation  = -30;   // deg from horizontal
  mjvOption opt;      mjv_defaultOption(&opt);
  mjvScene scene;     mjv_defaultScene(&scene);
  mjrContext con;     mjr_defaultContext(&con);
  static AppCtx app{m, &scene, &cam};
  glfwSetWindowUserPointer(win, &app);
  glfwSetMouseButtonCallback(win, mouse_button_cb);
  glfwSetCursorPosCallback(win, cursor_pos_cb);
  glfwSetScrollCallback(win, scroll_cb);
  mjv_makeScene(m, &scene, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  int hid = mj_name2id(m, mjOBJ_HFIELD, "terrain");

  
  upload_heightfield(m, hid, &con);
  

  // ---- ROS2
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MultiCamNode>(m, d, hid, &con);
  std::thread ros_thread([&](){ rclcpp::spin(node); });

  // Render / step loop
  while (!glfwWindowShouldClose(win)) {
    // Step physics lightly
    mj_step(m, d);
    std::vector<float> staged;
    if (node->pop_staged(staged)) {
      set_heightfield(m, hid, staged);   // write to model buffer
      mjr_uploadHField(m, &con, hid);    // upload to GPU (GL thread)
    }
    // Render
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(win, &viewport.width, &viewport.height);
    
    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scene);
    mjr_render(viewport, &scene, &con);

    glfwSwapBuffers(win);
    glfwPollEvents();
  }

  // Cleanup
  rclcpp::shutdown();
  ros_thread.join();

  mjr_freeContext(&con);
  mjv_freeScene(&scene);
  glfwDestroyWindow(win);
  glfwTerminate();

  mj_deleteData(d);
  mj_deleteModel(m);

  return 0;
}
