cp $SRC/cpp-shuda/eigen/gradient_gtest/Gradients.hpp ./Gradients.hpp
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared gvnn.cc -o gvnn.so -fPIC -I $TF_INC -I $SOPHUS_INC -I /usr/local/include/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
