import torch
import time
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt


GREEN = '\033[92m'

EXPORT_MODEL_FOLDER = "models"
BATCH_SIZE = 2
NB_ITERATIONS = 100

import os
if not os.path.exists(EXPORT_MODEL_FOLDER):
    os.makedirs(EXPORT_MODEL_FOLDER)

# Load the model.
vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
vitb16.eval()

# Export the model
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
torch.onnx.export(vitb16,
                dummy_input, # example input.
                EXPORT_MODEL_FOLDER + "/dino_vit.onnx", # output file.
                opset_version=12, # ONNX version.
                input_names=['x.1'], # input name.
                output_names=['output'], # output name.
                dynamic_axes={'x.1': {0: 'batch_size'}, # first dim of the input tensor can vary in size and will be named 'batch_size'.
                            'output': {0: 'batch_size'}}) # same here too.


# Run inference PyTorch CPU.
print("Benchmarking PyTorch CPU...")
device = torch.device('cpu')
vitb16_on_cpu = vitb16.to(device)
vitb16_on_cpu.eval()

times_list = []
for i in range(NB_ITERATIONS):
    with torch.no_grad():
        # Fake data.
        dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
        dummy_input = dummy_input.to(device)
        
        time_start = time.time()
        output = vitb16_on_cpu(dummy_input)
        time_end = time.time()
        times_list.append(time_end - time_start)

# Print the average time taken for inference
avg_time_pytorch_cpu = sum(times_list) / len(times_list)
print("     Average time taken for inference on CPU:", avg_time_pytorch_cpu)



# Run inference PyTorch GPU.
print("Benchmarking PyTorch GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device == torch.device('cuda')

vitb16_on_gpu = vitb16.to(device)
vitb16_on_gpu.eval()

# Run inference PyTorch GPU.
times_list = []
for i in range(NB_ITERATIONS):
    with torch.no_grad():
        dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
        dummy_input = dummy_input.to(device)

        time_start = time.time()
        output = vitb16_on_gpu(dummy_input)
        time_end = time.time()
        times_list.append(time_end - time_start)

# Print the average time taken for inference
avg_time_pytorch_gpu = sum(times_list) / len(times_list)
print("     Average time taken for inference on GPU:", avg_time_pytorch_gpu)



# Do the same benchmarking with the exported model in ONNX format.
print("Benchmarking ONNX model with the CUDAExecutionProvider backend...")
# Check sessions.
assert 'CUDAExecutionProvider' in ort.get_available_providers() # check if CUDA is available.
print(ort.get_available_providers())

ort_session = ort.InferenceSession(EXPORT_MODEL_FOLDER + "/dino_vit.onnx", providers=["CUDAExecutionProvider"])

times_list = []
for i in range(NB_ITERATIONS):
    with torch.no_grad():
        dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
        # Convert to numpy (must be on CPU, but ONNX Runtime will transfer to GPU internally)
        dummy_input_np = dummy_input.detach().cpu().numpy() # NOTE: this is a copy of the data, not great, there's an alternative way to do this using io_bindings

        time_start = time.time()
        output = ort_session.run(None, # None refers to return all output tensors defined in the model.
                                 {"x.1": dummy_input_np})
        time_end = time.time()
        times_list.append(time_end - time_start)

# Print the average time taken for inference.
avg_time_onnx_cuda = sum(times_list) / len(times_list)
print("     Average time taken for inference with ONNX model:" + GREEN + str(avg_time_onnx_cuda) + '\033[0m')


# Do the same benchmarking with the exported model in ONNX format using IO bindings.
# IO bindings provide a way to directly connect your application's memory
# (including GPU memory) to ONNX Runtime's memory space without unnecessary
# copies.

# Do the same benchmarking with the exported model.
print("Benchmarking ONNX model with the TensorrtExecutionProvider backend...")
import onnxruntime as ort
# Check sessions.
assert 'TensorrtExecutionProvider' in ort.get_available_providers() # check if CUDA is available.
ort_session = ort.InferenceSession(EXPORT_MODEL_FOLDER + "/dino_vit.onnx", providers=["CUDAExecutionProvider"])
# Run inference

times_list = []
for i in range(NB_ITERATIONS):
    with torch.no_grad():
        # Create new input for each iteration
        dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
        # Convert to numpy (must be on CPU, but ONNX Runtime will transfer to GPU internally)
        dummy_input_np = dummy_input.detach().cpu().numpy()

        time_start = time.time()
        output = ort_session.run(None, # None refers to return all output tensors defined in the model.
                                 {"x.1": dummy_input_np})
        time_end = time.time()
        times_list.append(time_end - time_start)
# Print the average time taken for inference
avg_time_onnx_tensorrt = sum(times_list) / len(times_list)
print("     Average time taken for inference with ONNX model:" + GREEN + str(avg_time_onnx_tensorrt) + '\033[0m')


# Plot the average time.
labels = ['PyTorch GPU', 'ONNX CUDAExecutionProvider', 'ONNX TensorrtExecutionProvider']
times = [avg_time_pytorch_gpu, avg_time_onnx_cuda, avg_time_onnx_tensorrt]

plt.figure(figsize=(10, 6))
plt.bar(labels, times, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Average Time (s)')
plt.title('Average Time for Inference on CPU, GPU, and ONNX Model')
plt.savefig('benchmarking_dino.png')

# Verify that the outputs are the same for the 3 models.
for i in range(NB_ITERATIONS):
    with torch.no_grad():
        dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
        output_pytorch_cpu: torch.Tensor = vitb16_on_cpu(dummy_input)
        output_pytorch_gpu: torch.Tensor = vitb16_on_gpu(dummy_input)
        output_onnx: list[torch.Tensor] = ort_session.run(None, {"x.1": dummy_input.detach().cpu().numpy()})

        assert np.allclose(output_pytorch_cpu.detach().cpu().numpy(), output_onnx[0], atol=1e-2), "The outputs of the PyTorch CPU and ONNX models are not the same."
        assert np.allclose(output_pytorch_gpu.detach().cpu().numpy(), output_onnx[0], atol=1e-2), "The outputs of the PyTorch GPU and ONNX models are not the same."
        assert torch.allclose(output_pytorch_cpu, output_pytorch_gpu), "The outputs of the PyTorch CPU and GPU models are not the same."
        


