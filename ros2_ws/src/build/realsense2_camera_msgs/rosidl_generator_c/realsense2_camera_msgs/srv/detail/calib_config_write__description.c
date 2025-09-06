// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from realsense2_camera_msgs:srv/CalibConfigWrite.idl
// generated code does not contain a copyright notice

#include "realsense2_camera_msgs/srv/detail/calib_config_write__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_realsense2_camera_msgs
const rosidl_type_hash_t *
realsense2_camera_msgs__srv__CalibConfigWrite__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x1f, 0x03, 0x25, 0x63, 0x0f, 0xac, 0x02, 0x7d,
      0x65, 0x8f, 0x69, 0x1e, 0x26, 0x90, 0xd5, 0x20,
      0xf4, 0x89, 0x93, 0xd8, 0x46, 0x13, 0x06, 0x6c,
      0x44, 0xe6, 0x31, 0x5d, 0xf9, 0xa8, 0xce, 0xd3,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_realsense2_camera_msgs
const rosidl_type_hash_t *
realsense2_camera_msgs__srv__CalibConfigWrite_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x0c, 0xef, 0xc2, 0x3f, 0x3b, 0x1d, 0x71, 0x84,
      0xeb, 0x83, 0x98, 0x10, 0x95, 0xe4, 0x3b, 0x55,
      0x21, 0xea, 0x99, 0xfd, 0xee, 0x28, 0x55, 0x3a,
      0x3e, 0x63, 0x99, 0xf2, 0x72, 0x26, 0x59, 0xe2,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_realsense2_camera_msgs
const rosidl_type_hash_t *
realsense2_camera_msgs__srv__CalibConfigWrite_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xe0, 0xcc, 0x5e, 0xec, 0xb8, 0x06, 0xfa, 0xa1,
      0xa2, 0x55, 0xa7, 0xf3, 0x83, 0x8f, 0x29, 0xd2,
      0x2c, 0x7c, 0xc2, 0x71, 0xd2, 0x88, 0x71, 0xe4,
      0x6c, 0x94, 0x23, 0x47, 0x71, 0xdd, 0x8c, 0x2e,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_realsense2_camera_msgs
const rosidl_type_hash_t *
realsense2_camera_msgs__srv__CalibConfigWrite_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x79, 0x85, 0xe8, 0xf3, 0xe6, 0xa3, 0x60, 0xd9,
      0xdb, 0xff, 0xc3, 0xe3, 0xd9, 0x99, 0x68, 0x8e,
      0x34, 0x0d, 0x3d, 0x79, 0xe3, 0x86, 0x7c, 0x57,
      0x8c, 0xa1, 0xae, 0x75, 0x81, 0x38, 0x59, 0x02,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "service_msgs/msg/detail/service_event_info__functions.h"
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t service_msgs__msg__ServiceEventInfo__EXPECTED_HASH = {1, {
    0x41, 0xbc, 0xbb, 0xe0, 0x7a, 0x75, 0xc9, 0xb5,
    0x2b, 0xc9, 0x6b, 0xfd, 0x5c, 0x24, 0xd7, 0xf0,
    0xfc, 0x0a, 0x08, 0xc0, 0xcb, 0x79, 0x21, 0xb3,
    0x37, 0x3c, 0x57, 0x32, 0x34, 0x5a, 0x6f, 0x45,
  }};
#endif

static char realsense2_camera_msgs__srv__CalibConfigWrite__TYPE_NAME[] = "realsense2_camera_msgs/srv/CalibConfigWrite";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char realsense2_camera_msgs__srv__CalibConfigWrite_Event__TYPE_NAME[] = "realsense2_camera_msgs/srv/CalibConfigWrite_Event";
static char realsense2_camera_msgs__srv__CalibConfigWrite_Request__TYPE_NAME[] = "realsense2_camera_msgs/srv/CalibConfigWrite_Request";
static char realsense2_camera_msgs__srv__CalibConfigWrite_Response__TYPE_NAME[] = "realsense2_camera_msgs/srv/CalibConfigWrite_Response";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";

// Define type names, field names, and default values
static char realsense2_camera_msgs__srv__CalibConfigWrite__FIELD_NAME__request_message[] = "request_message";
static char realsense2_camera_msgs__srv__CalibConfigWrite__FIELD_NAME__response_message[] = "response_message";
static char realsense2_camera_msgs__srv__CalibConfigWrite__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field realsense2_camera_msgs__srv__CalibConfigWrite__FIELDS[] = {
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {realsense2_camera_msgs__srv__CalibConfigWrite_Request__TYPE_NAME, 51, 51},
    },
    {NULL, 0, 0},
  },
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {realsense2_camera_msgs__srv__CalibConfigWrite_Response__TYPE_NAME, 52, 52},
    },
    {NULL, 0, 0},
  },
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {realsense2_camera_msgs__srv__CalibConfigWrite_Event__TYPE_NAME, 49, 49},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription realsense2_camera_msgs__srv__CalibConfigWrite__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Event__TYPE_NAME, 49, 49},
    {NULL, 0, 0},
  },
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Request__TYPE_NAME, 51, 51},
    {NULL, 0, 0},
  },
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Response__TYPE_NAME, 52, 52},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
realsense2_camera_msgs__srv__CalibConfigWrite__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {realsense2_camera_msgs__srv__CalibConfigWrite__TYPE_NAME, 43, 43},
      {realsense2_camera_msgs__srv__CalibConfigWrite__FIELDS, 3, 3},
    },
    {realsense2_camera_msgs__srv__CalibConfigWrite__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = realsense2_camera_msgs__srv__CalibConfigWrite_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = realsense2_camera_msgs__srv__CalibConfigWrite_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = realsense2_camera_msgs__srv__CalibConfigWrite_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char realsense2_camera_msgs__srv__CalibConfigWrite_Request__FIELD_NAME__calib_config[] = "calib_config";

static rosidl_runtime_c__type_description__Field realsense2_camera_msgs__srv__CalibConfigWrite_Request__FIELDS[] = {
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Request__FIELD_NAME__calib_config, 12, 12},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
realsense2_camera_msgs__srv__CalibConfigWrite_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {realsense2_camera_msgs__srv__CalibConfigWrite_Request__TYPE_NAME, 51, 51},
      {realsense2_camera_msgs__srv__CalibConfigWrite_Request__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char realsense2_camera_msgs__srv__CalibConfigWrite_Response__FIELD_NAME__success[] = "success";
static char realsense2_camera_msgs__srv__CalibConfigWrite_Response__FIELD_NAME__error_message[] = "error_message";

static rosidl_runtime_c__type_description__Field realsense2_camera_msgs__srv__CalibConfigWrite_Response__FIELDS[] = {
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Response__FIELD_NAME__error_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
realsense2_camera_msgs__srv__CalibConfigWrite_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {realsense2_camera_msgs__srv__CalibConfigWrite_Response__TYPE_NAME, 52, 52},
      {realsense2_camera_msgs__srv__CalibConfigWrite_Response__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char realsense2_camera_msgs__srv__CalibConfigWrite_Event__FIELD_NAME__info[] = "info";
static char realsense2_camera_msgs__srv__CalibConfigWrite_Event__FIELD_NAME__request[] = "request";
static char realsense2_camera_msgs__srv__CalibConfigWrite_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field realsense2_camera_msgs__srv__CalibConfigWrite_Event__FIELDS[] = {
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {realsense2_camera_msgs__srv__CalibConfigWrite_Request__TYPE_NAME, 51, 51},
    },
    {NULL, 0, 0},
  },
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {realsense2_camera_msgs__srv__CalibConfigWrite_Response__TYPE_NAME, 52, 52},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription realsense2_camera_msgs__srv__CalibConfigWrite_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Request__TYPE_NAME, 51, 51},
    {NULL, 0, 0},
  },
  {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Response__TYPE_NAME, 52, 52},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
realsense2_camera_msgs__srv__CalibConfigWrite_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {realsense2_camera_msgs__srv__CalibConfigWrite_Event__TYPE_NAME, 49, 49},
      {realsense2_camera_msgs__srv__CalibConfigWrite_Event__FIELDS, 3, 3},
    },
    {realsense2_camera_msgs__srv__CalibConfigWrite_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = realsense2_camera_msgs__srv__CalibConfigWrite_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = realsense2_camera_msgs__srv__CalibConfigWrite_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "string calib_config\n"
  "---\n"
  "bool success\n"
  "string error_message";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
realsense2_camera_msgs__srv__CalibConfigWrite__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {realsense2_camera_msgs__srv__CalibConfigWrite__TYPE_NAME, 43, 43},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 58, 58},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
realsense2_camera_msgs__srv__CalibConfigWrite_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Request__TYPE_NAME, 51, 51},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
realsense2_camera_msgs__srv__CalibConfigWrite_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Response__TYPE_NAME, 52, 52},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
realsense2_camera_msgs__srv__CalibConfigWrite_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {realsense2_camera_msgs__srv__CalibConfigWrite_Event__TYPE_NAME, 49, 49},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
realsense2_camera_msgs__srv__CalibConfigWrite__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *realsense2_camera_msgs__srv__CalibConfigWrite__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *realsense2_camera_msgs__srv__CalibConfigWrite_Event__get_individual_type_description_source(NULL);
    sources[3] = *realsense2_camera_msgs__srv__CalibConfigWrite_Request__get_individual_type_description_source(NULL);
    sources[4] = *realsense2_camera_msgs__srv__CalibConfigWrite_Response__get_individual_type_description_source(NULL);
    sources[5] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
realsense2_camera_msgs__srv__CalibConfigWrite_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *realsense2_camera_msgs__srv__CalibConfigWrite_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
realsense2_camera_msgs__srv__CalibConfigWrite_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *realsense2_camera_msgs__srv__CalibConfigWrite_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
realsense2_camera_msgs__srv__CalibConfigWrite_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *realsense2_camera_msgs__srv__CalibConfigWrite_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *realsense2_camera_msgs__srv__CalibConfigWrite_Request__get_individual_type_description_source(NULL);
    sources[3] = *realsense2_camera_msgs__srv__CalibConfigWrite_Response__get_individual_type_description_source(NULL);
    sources[4] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
