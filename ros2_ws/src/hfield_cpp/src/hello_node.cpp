#include <rclcpp/rclcpp.hpp>

class HelloNode : public rclcpp::Node {
public:
  HelloNode() : Node("hello_node") {
    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      [this]() { RCLCPP_INFO(this->get_logger(), "Hello from C++!"); });
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HelloNode>());
  rclcpp::shutdown();
  return 0;
}
