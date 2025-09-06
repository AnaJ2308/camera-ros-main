// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from realsense2_camera_msgs:srv/CalibConfigWrite.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "realsense2_camera_msgs/srv/calib_config_write.h"


#ifndef REALSENSE2_CAMERA_MSGS__SRV__DETAIL__CALIB_CONFIG_WRITE__STRUCT_H_
#define REALSENSE2_CAMERA_MSGS__SRV__DETAIL__CALIB_CONFIG_WRITE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'calib_config'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/CalibConfigWrite in the package realsense2_camera_msgs.
typedef struct realsense2_camera_msgs__srv__CalibConfigWrite_Request
{
  rosidl_runtime_c__String calib_config;
} realsense2_camera_msgs__srv__CalibConfigWrite_Request;

// Struct for a sequence of realsense2_camera_msgs__srv__CalibConfigWrite_Request.
typedef struct realsense2_camera_msgs__srv__CalibConfigWrite_Request__Sequence
{
  realsense2_camera_msgs__srv__CalibConfigWrite_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} realsense2_camera_msgs__srv__CalibConfigWrite_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'error_message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/CalibConfigWrite in the package realsense2_camera_msgs.
typedef struct realsense2_camera_msgs__srv__CalibConfigWrite_Response
{
  bool success;
  rosidl_runtime_c__String error_message;
} realsense2_camera_msgs__srv__CalibConfigWrite_Response;

// Struct for a sequence of realsense2_camera_msgs__srv__CalibConfigWrite_Response.
typedef struct realsense2_camera_msgs__srv__CalibConfigWrite_Response__Sequence
{
  realsense2_camera_msgs__srv__CalibConfigWrite_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} realsense2_camera_msgs__srv__CalibConfigWrite_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  realsense2_camera_msgs__srv__CalibConfigWrite_Event__request__MAX_SIZE = 1
};
// response
enum
{
  realsense2_camera_msgs__srv__CalibConfigWrite_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/CalibConfigWrite in the package realsense2_camera_msgs.
typedef struct realsense2_camera_msgs__srv__CalibConfigWrite_Event
{
  service_msgs__msg__ServiceEventInfo info;
  realsense2_camera_msgs__srv__CalibConfigWrite_Request__Sequence request;
  realsense2_camera_msgs__srv__CalibConfigWrite_Response__Sequence response;
} realsense2_camera_msgs__srv__CalibConfigWrite_Event;

// Struct for a sequence of realsense2_camera_msgs__srv__CalibConfigWrite_Event.
typedef struct realsense2_camera_msgs__srv__CalibConfigWrite_Event__Sequence
{
  realsense2_camera_msgs__srv__CalibConfigWrite_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} realsense2_camera_msgs__srv__CalibConfigWrite_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // REALSENSE2_CAMERA_MSGS__SRV__DETAIL__CALIB_CONFIG_WRITE__STRUCT_H_
