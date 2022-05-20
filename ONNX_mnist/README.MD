Getting Started: 
- Clone / Download to your device
- Install neccessary library to your environment: tensorflow + pytorch (recommend: GPU version), onnx, onnxruntime-gpu, ...

(onnx-tensorflow: we can clone here: https://github.com/onnx/onnx-tensorflow, then copy/move onnx_tf folder out of this folder)


DETAILS:

* train.ipynb: This is the source to create the Pytorch's model (here is MNIST)                   -----output----->          folder[data/MNIST/raw] + [mnist.pt]

* pt_2_ONNX.ipynb: This is the source to transform from Pytorch model to ONNX model               -----output----->          [mnist.onnx]

* ONNX_2_tf. ipynb: This is the source to transform from ONNX model to Tensorflow.Keras model     -----output----->          folder[mnist] + [mnist.h5]

* 3.jpg: This is the test image

* ONNX.pptx: This is our report slide show.
