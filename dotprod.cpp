#include <stdio.h>  // Include the standard I/O library
#include <stdlib.h>  // Include the standard library
#include <CL/cl.h>  // Include the OpenCL library
#include <chrono>  // Include the chrono library for timing

#define PRINT 1  // Define a constant to control the printing of vectors

int SZ = 100000000;  // Set the size of the vectors

int *v1, *v2, *v_out;  // Declare pointers for the input and output vectors

cl_mem bufV1, bufV2, bufV_out;  // Declare OpenCL memory buffers for the vectors

cl_device_id device_id;  // Declare an OpenCL device ID
cl_context context;  // Declare an OpenCL context
cl_program program;  // Declare an OpenCL program

cl_kernel kernel;  // Declare an OpenCL kernel
cl_command_queue queue;  // Declare an OpenCL command queue

cl_event event = NULL;  // Declare an OpenCL event

int err;  // Declare an error variable

cl_device_id create_device();  // Declare a function to create an OpenCL device
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);  // Declare a function to set up the OpenCL device, context, queue, and kernel
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);  // Declare a function to build an OpenCL program
void setup_kernel_memory();  // Declare a function to set up the kernel memory
void copy_kernel_args();  // Declare a function to copy the kernel arguments
void free_memory();  // Declare a function to free the allocated memory
void init(int *&A, int size);  // Declare a function to initialize a vector
void print(int *A, int size);  // Declare a function to print a vector

int main(int argc, char **argv) {
    // Check if a command-line argument was provided to set the vector size
    if (argc > 1) {
        SZ = atoi(argv[1]);
    }

    // Initialize the input and output vectors
    init(v1, SZ);
    init(v2, SZ);
    init(v_out, SZ);

    // Set the global work size for the OpenCL kernel
    size_t global[1] = {(size_t)SZ};

    // Print the input vectors
    print(v1, SZ);
    print(v2, SZ);

    // Set up the OpenCL device, context, queue, and kernel
    setup_openCL_device_context_queue_kernel((char *)"./vector_ops_ocl.cl", (char *)"vector_add_ocl");

    // Set up the kernel memory
    setup_kernel_memory();

    // Copy the kernel arguments
    copy_kernel_args();

    // Get the start time for the kernel execution
    auto start = std::chrono::high_resolution_clock::now();

    // Enqueue the kernel and wait for it to complete
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event);

    // Read the output vector from the device
    clEnqueueReadBuffer(queue, bufV_out, CL_TRUE, 0, SZ * sizeof(int), &v_out[0], 0, NULL, NULL);

    // Print the output vector
    print(v_out, SZ);

    // Get the end time and calculate the kernel execution time
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = stop - start;
    printf("Kernel Execution Time: %f ms\n", elapsed_time.count());

    // Free the allocated memory
    free_memory();

    return 0;
}

// Function to initialize a vector with random values
void init(int *&A, int size) {
    A = (int *)malloc(sizeof(int) * size);

    for (long i = 0; i < size; i++) {
        A[i] = rand() % 100;
    }
}

// Function to print a vector
void print(int *A, int size) {
    if (PRINT == 0) {
        return;
    }

    if (PRINT == 1 && size > 15) {
        for (long i = 0; i < 5; i++) {
            printf("%d ", A[i]);
        }
        printf(" ..... ");
        for (long i = size - 5; i < size; i++) {
            printf("%d ", A[i]);
        }
    } else {
        for (long i = 0; i < size; i++) {
            printf("%d ", A[i]);
        }
    }
    printf("\n----------------------------\n");
}

// Function to free the allocated memory
void free_memory() {
    clReleaseMemObject(bufV1);
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV_out);

    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(v1);
    free(v2);
    free(v_out);
}

// Function to copy the kernel arguments
void copy_kernel_args() {
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufV_out);

    if (err < 0) {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

// Function to set up the kernel memory
void setup_kernel_memory() {
    bufV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV_out = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0, NULL, NULL);
}

// Function to set up the OpenCL device, context, queue, and kernel
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname) {
    device_id = create_device();
    cl_int err;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device_id, filename);

    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    };

    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0) {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    };
}

// Function to build an OpenCL program
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename) {
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    program_handle = fopen(filename, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

// Function to create an OpenCL device
cl_device_id create_device() {
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        perror("Couldn't identify a platform");
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        printf("GPU not found\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0) {
        perror("Couldn't access any devices");
        exit(1);
    }

    return dev;
}