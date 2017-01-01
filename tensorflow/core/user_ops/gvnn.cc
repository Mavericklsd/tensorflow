#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <so3.hpp>
#include <se3.hpp>
#include <sim3.hpp>
#include <Eigen/Dense>
#include "Gradients.hpp"

using namespace tensorflow;
using namespace Sophus;
using namespace Eigen;

namespace tensorflow{

using namespace tensorflow;

class SO3Op : public OpKernel {
 public:
  explicit SO3Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& rotation_vector = context->input(0);
    auto rot_vec = rotation_vector.flat<float>();
    printf("Debug SO3 rotation_vector: %s \n",rotation_vector.DebugString().c_str());

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({3, 3}), &output_tensor));
    auto rot_mat = output_tensor->flat<float>();

    // const int N = rot_vec.size();
    Matrix<float,3,1> vec(rot_vec(0),rot_vec(1),rot_vec(2));
    SO3Group<float> RR = SO3Group<float>::exp(vec);
    Matrix<float,3,3> R = RR.matrix().transpose();
    for (int i = 0; i < 9; i++) {
      rot_mat(i) = R(i);
    }

    printf("Debug SO3 Output: %s \n",output_tensor->DebugString().c_str());
    return;
  }
};

REGISTER_KERNEL_BUILDER(Name("So3").Device(DEVICE_CPU), SO3Op);
REGISTER_OP("So3")
    .Attr("T: realnumbertype")
    .Input("rotation_vector: T")
    .Output("rotation_matrix: T")
    // .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    //   c->set_output(0, c->input(0));
    //   return Status::OK();
    // })
    ;

class SO3GradOp: public OpKernel {
public:
    explicit SO3GradOp(OpKernelConstruction* context) :
            OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
        printf("called SO3GradOp.Compute() \n");
        const Tensor& rotation_vector = context->input(0);  //1x3
        const Tensor& gradients_tensor = context->input(1); //1x9
        printf("Debug SO3GradOp Features Input: %s \n",rotation_vector.DebugString().c_str());
        printf("Debug SO3GradOp Gradients Input: %s \n",gradients_tensor.DebugString().c_str());

        TensorShape output_shape = rotation_vector.shape();

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context,context->allocate_output(0, output_shape, &output_tensor));
        output_tensor->flat<float>().setZero();

        auto rot_vec = rotation_vector.flat<float>();
        Matrix<float,3,1> vec(rot_vec(0),rot_vec(1),rot_vec(2));
        Matrix<float,9,3> dSO3 = dso3_by_dv(vec,1e-12f);

        const float* grad = gradients_tensor.flat<float>().data();
        Matrix<float,1,9> row_grad; 
        row_grad << grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7], grad[8];
        Matrix<float,1,3> row_out;
        row_out = row_grad * dSO3;  
        float* loss_div_rot_vec = output_tensor->flat<float>().data();
        loss_div_rot_vec[0] = row_out[0];
        loss_div_rot_vec[1] = row_out[1];
        loss_div_rot_vec[2] = row_out[2];

        printf("Debug SO3GradOp Output: %s \n",output_tensor->DebugString().c_str());
        printf("---------------------------------- \n");
        return;
    }
};

REGISTER_KERNEL_BUILDER(Name("So3Grad").Device(DEVICE_CPU), SO3GradOp);

REGISTER_OP("So3Grad")
.Attr("T: realnumbertype")
.Input("rotation_vector: T")
.Input("loss_div_rot_matrix: T")
.Output("loss_div_rot_vector: T")
.Doc(R"doc(
TODO!!
)doc");


class SE3Op : public OpKernel {
 public:
  explicit SE3Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({4, 4}),
                                                     &output_tensor));
    auto output = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    Matrix< float, 6, 1> tangent; 
    tangent << input(0),input(1),input(2),input(3),input(4),input(5);
    Matrix4f T =  SE3Group<float>::exp(tangent).matrix().transpose();

    for (int i = 0; i < 16; i++) {
      output(i) = T(i);
    }
    return;
  }
};

REGISTER_KERNEL_BUILDER(Name("Se3").Device(DEVICE_CPU), SE3Op);

REGISTER_OP("Se3")
    .Input("motion_vector: float")
    .Output("transformation_matrix: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class SE3GradOp: public OpKernel {
public:
    explicit SE3GradOp(OpKernelConstruction* context) :
            OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
        printf("called SE3GradOp.Compute() \n");
        const Tensor& se3_vector = context->input(0);  //1x3
        const Tensor& gradients_tensor = context->input(1); //1x9
        printf("Debug SE3GradOp Features Input: %s \n",se3_vector.DebugString().c_str());
        printf("Debug SE3GradOp Gradients Input: %s \n",gradients_tensor.DebugString().c_str());

        TensorShape output_shape = se3_vector.shape();

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context,context->allocate_output(0, output_shape, &output_tensor));
        output_tensor->flat<float>().setZero();

        auto se3_vec = se3_vector.flat<float>();
        Matrix<float,6,1> vec; vec << se3_vec(0),se3_vec(1),se3_vec(2),se3_vec(3),se3_vec(4),se3_vec(5);
        Matrix<float,12,6> dSE3 = dse3_by_dv(vec,1e-12f);

        const float* grad = gradients_tensor.flat<float>().data();
        Matrix<float,1,12> row_grad; 
        row_grad << grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7], grad[8], grad[9], grad[10], grad[11];
        Matrix<float,1,6> row_out;
        row_out = row_grad * dSE3;  
        float* loss_div_rot_vec = output_tensor->flat<float>().data();
        loss_div_rot_vec[0] = row_out[0];
        loss_div_rot_vec[1] = row_out[1];
        loss_div_rot_vec[2] = row_out[2];
        loss_div_rot_vec[3] = row_out[3];
        loss_div_rot_vec[4] = row_out[4];
        loss_div_rot_vec[5] = row_out[5];

        printf("Debug SE3GradOp Output: %s \n",output_tensor->DebugString().c_str());
        printf("---------------------------------- \n");
        return;
    }
};

REGISTER_KERNEL_BUILDER(Name("Se3Grad").Device(DEVICE_CPU), SO3GradOp);

REGISTER_OP("Se3Grad")
.Attr("T: realnumbertype")
.Input("se3_vector: T")
.Input("loss_div_se3_matrix: T")
.Output("loss_div_se3_vector: T")
.Doc(R"doc(
TODO!!
)doc");

class Sim3Op : public OpKernel {
 public:
  explicit Sim3Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    printf("called Sim3Op.Compute() \n");
    const Tensor& sim3_vector = context->input(0);  //1x3
    printf("Debug Sim3Op Features Input: %s \n",sim3_vector.DebugString().c_str());

    // Create an output tensor      
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({4, 4}),&output_tensor));
    auto output = output_tensor->flat<float>();
    output_tensor->flat<float>().setZero();

    // Grab the input tensor
    auto sim3_vec = sim3_vector.flat<float>();
    Matrix<float,7,1> vec; vec << sim3_vec(0),sim3_vec(1),sim3_vec(2),sim3_vec(3),sim3_vec(4),sim3_vec(5),sim3_vec(6);

    // Set all but the first element of the output tensor to 0.
    // const int N = input.size();
    Matrix4f T = Sim3Group<float>::exp(vec).matrix().transpose();
    int i=0;
    for (int i = 0; i < 16; i++) {
      output(i) = T(i);
    }
    return;
  }
};

REGISTER_KERNEL_BUILDER(Name("Sim3").Device(DEVICE_CPU), Sim3Op);

REGISTER_OP("Sim3")
    .Input("sim3_vector: float")
    .Output("sim3_matrix: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class Sim3GradOp: public OpKernel {
public:
    explicit Sim3GradOp(OpKernelConstruction* context) :
            OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
        printf("called Sim3GradOp.Compute() \n");
        const Tensor& sim3_vector = context->input(0);  //1x3
        const Tensor& gradients_tensor = context->input(1); //1x9
        printf("Debug Sim3GradOp Features Input: %s \n",sim3_vector.DebugString().c_str());
        printf("Debug Sim3GradOp Gradients Input: %s \n",gradients_tensor.DebugString().c_str());

        TensorShape output_shape = sim3_vector.shape();

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context,context->allocate_output(0, output_shape, &output_tensor));
        output_tensor->flat<float>().setZero();

        auto sim3_vec = sim3_vector.flat<float>();
        Matrix<float,7,1> vec; vec << sim3_vec(0),sim3_vec(1),sim3_vec(2),sim3_vec(3),sim3_vec(4),sim3_vec(5),sim3_vec(6);
        Matrix<float,12,7> dSE3 = dsim3_by_dv_numerical(vec,1e-12f);

        const float* grad = gradients_tensor.flat<float>().data();
        Matrix<float,1,12> row_grad; 
        row_grad << grad[0], grad[1], grad[2], grad[3], grad[4], grad[5], grad[6], grad[7], grad[8], grad[9], grad[10], grad[11];
        Matrix<float,1,7> row_out;
        row_out = row_grad * dSE3;  
        float* loss_div_rot_vec = output_tensor->flat<float>().data();
        loss_div_rot_vec[0] = row_out[0];
        loss_div_rot_vec[1] = row_out[1];
        loss_div_rot_vec[2] = row_out[2];
        loss_div_rot_vec[3] = row_out[3];
        loss_div_rot_vec[4] = row_out[4];
        loss_div_rot_vec[5] = row_out[5];
        loss_div_rot_vec[6] = row_out[6];

        printf("Debug Sim3GradOp Output: %s \n",output_tensor->DebugString().c_str());
        printf("---------------------------------- \n");
        return;
    }
};

REGISTER_KERNEL_BUILDER(Name("Sim3Grad").Device(DEVICE_CPU), Sim3GradOp);

REGISTER_OP("Sim3Grad")
.Attr("T: realnumbertype")
.Input("sim3_vector: T")
.Input("loss_div_sim3_matrix: T")
.Output("loss_div_sim3_vector: T")
.Doc(R"doc(
TODO!!
)doc");

class ProjLayerOp : public OpKernel {
 public:
  explicit ProjLayerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("intrinsics", &_instrinsics));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({2, 1}),
                                                     &output_tensor));
    auto output = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    output(0) = _instrinsics[0]*input(0)/input(2)+_instrinsics[2];
    output(1) = _instrinsics[1]*input(1)/input(2)+_instrinsics[2];
  }

  Tensor tensor_;
  TF_DISALLOW_COPY_AND_ASSIGN(ProjLayerOp);
  std::vector<float> _instrinsics;
};

REGISTER_KERNEL_BUILDER(Name("ProjLayer").Device(DEVICE_CPU), ProjLayerOp);

REGISTER_OP("ProjLayer")
    .Attr("intrinsics: list(float) >= 4")
    .Input("motion_vector: float")
    .Output("transformation_matrix: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


template<typename Device, typename T>
class MyCopyOp: public OpKernel {
public:
    explicit MyCopyOp(OpKernelConstruction* context) :
            OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        auto in_flat = input.flat<T>();

        printf("Debug MyCopyOp Features: %s \n",input.DebugString().c_str());

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                context->allocate_output(0, input.shape(), &output));

        auto out_flat = output->flat<T>();
        out_flat.setZero();

        for (int d = 0; d < input.dims(); ++d) {
            for (int i = 0; i < input.dim_size(d); ++i) {
                out_flat(d * input.dim_size(d) + i) = in_flat(
                        d * input.dim_size(d) + i);
            }
        }

        printf("Debug MyCopyOp Output: %s \n",output->DebugString().c_str());
    }

};


template<typename Device, typename T>
class MyCopyGradOp: public OpKernel {
public:
    explicit MyCopyGradOp(OpKernelConstruction* context) :
            OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
        printf("called MyCopyGradOp.Compute() \n");
        const Tensor& gradients = context->input(0);
        const Tensor& features = context->input(1);
        printf("Debug MyCopyOpGrad Gradients: %s \n",gradients.DebugString().c_str());
        printf("Debug MyCopyOpGrad Features: %s \n",features.DebugString().c_str());

        TensorShape output_shape = features.shape();

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                context->allocate_output(0, output_shape, &output));
        output->flat<T>().setZero();

        const T* btm_ptr = gradients.flat<T>().data();
        T* top_ptr = output->flat<T>().data();

        for (int i = 0; i < gradients.NumElements(); ++i) {
            top_ptr[i] = btm_ptr[i];
        }

        printf("Debug MyCopyOpGrad Output: %s \n",output->DebugString().c_str());
        printf("---------------------------------- \n");
    }
};



REGISTER_OP("MyCopy")
.Attr("T: realnumbertype")
.Input("features: T")
.Output("output: T")
.Doc(R"doc(
Copies all input values to the output
)doc");

REGISTER_OP("MyCopyGrad")
.Attr("T: realnumbertype")
.Input("gradients: T")
.Input("features: T")
.Output("backprops: T")
.Doc(R"doc(
TODO!!
)doc");


#define REGISTER_MYCOPY_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("MyCopy").Device(DEVICE_CPU).TypeConstraint<type>("T"),              \
      MyCopyOp<Eigen::ThreadPoolDevice, type>);                                 \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("MyCopyGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      MyCopyGradOp<Eigen::ThreadPoolDevice, type>);                             //  \
  // REGISTER_KERNEL_BUILDER(                                                      \
  //     Name("MyCopy").Device(DEVICE_GPU).TypeConstraint<type>("T"),              \
  //     MyCopyOp<Eigen::GpuDevice, type>);                                        \
  // REGISTER_KERNEL_BUILDER(                                                      \
  //     Name("MyCopyGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),          \
  //     MyCopyGradOp<Eigen::GpuDevice, type>);                                

REGISTER_MYCOPY_KERNELS(float); 
REGISTER_MYCOPY_KERNELS(int);
REGISTER_MYCOPY_KERNELS(double);


}
