#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <sophus/so3.hpp>

using namespace tensorflow;
using namespace Sophus;
using namespace Eigen;

namespace tensorflow{

using namespace tensorflow;

class OptFlowOp : public OpKernel {
 public:
  explicit OptFlowOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({3, 3}),
                                                     &output_tensor));
    auto output = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    Vector3f vec(input(0),input(1),input(2));
    SO3Group<float> RR = SO3Group<float>::exp(vec);
    Matrix3f R = RR.matrix();
    for (int i = 0; i < 9; i++) {
      output(i) = R(i);
    }

    // Preserve the first input value if possible.
    //if (N > 0) output(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("OptFlow").Device(DEVICE_CPU), OptFlowOp);

REGISTER_OP("OptFlow")
    .Input("image_coordinates: float")
    .Output("transformed_corrdinates: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}


