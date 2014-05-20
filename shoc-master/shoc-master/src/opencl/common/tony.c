#include "support.h"
extern int gpu_offset;
extern int tony_device;
#define CEIL(n,d) (n/d+(int)(n%d!=0))
#define Round(n,d) (CEIL(n,d)*d)
#define MAX(a,b) (a>b?a:b)
#define Round_down(n,d) (((int)(n/d)-1)*d)
// Time that each OpenCL kernel in this collection were queued.
cl_ulong    *queuedTime;

// Time that each OpenCL kernel in this collection were submitted to the device.
cl_ulong    *submitTime;

// Time that each OpenCL kernel in this collection started running on the device.
cl_ulong    *startTime;

// Time that each OpenCL kernel in this collection finished running on the device.
cl_ulong    *endTime;

using namespace std;

void FillTimingInfo(cl_event *event, const int idx)
{
queuedTime = new cl_ulong[2];
submitTime = new cl_ulong[2];
startTime = new cl_ulong[2];
endTime = new cl_ulong[2];
    int sidx, eidx;
    if (idx == 2) {
        sidx = 0; eidx = 1;
    } else
        sidx = eidx = idx;
    for (int i=sidx ; i<=eidx ; ++i) {
        cl_int err;
        err = clGetEventProfilingInfo(event[i], CL_PROFILING_COMMAND_QUEUED,
                                      sizeof(cl_ulong), &queuedTime[i], NULL);
        CL_CHECK_ERROR(err);
        err = clGetEventProfilingInfo(event[i], CL_PROFILING_COMMAND_SUBMIT,
                                      sizeof(cl_ulong), &submitTime[i], NULL);
        CL_CHECK_ERROR(err);
        err = clGetEventProfilingInfo(event[i], CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &startTime[i], NULL);
        CL_CHECK_ERROR(err);
        err = clGetEventProfilingInfo(event[i], CL_PROFILING_COMMAND_END,
                                      sizeof(cl_ulong), &endTime[i], NULL);
        CL_CHECK_ERROR(err);
    }

}
cl_ulong QueueSubmitDelay(const int idx)
{
    return submitTime[idx] - queuedTime[idx];
}

cl_ulong SubmitStartDelay(const int idx)
{
    return startTime[idx] - submitTime[idx];
}

cl_ulong SubmitEndRuntime(const int idx)
{
    return endTime[idx] - submitTime[idx];
}

cl_ulong StartEndRuntime(const int idx)
{
    return endTime[idx] - startTime[idx];
}

cl_ulong FullOverheadRuntime(const int idx)
{
    return endTime[idx] - queuedTime[idx];
}

void Event_Print(std::ostream &out,cl_event * event, const int idx)
{
    int sidx, eidx;
    if (idx == 2) {
        sidx = 0; eidx = 1;
    } else
        sidx = eidx = idx;
        for (int i=sidx ; i<=eidx ; ++i) {
            out << "--> Event id=" << event[i] << ": " << i << " <--" << endl;
            out << "  raw queuedTime ns = " << queuedTime[i] << endl;
            out << "  raw submitTime ns = " << submitTime[i] << endl;
            out << "  raw startTime ns  = " << startTime[i] << endl;
            out << "  raw endTime ns    = " << endTime[i] << endl;
            
            out << "  queued-submit delay  = " << QueueSubmitDelay(i)/1.e6    << " ms\n";
            out << "  submit-start delay   = " << SubmitStartDelay(i)/1.e6    << " ms\n";
            out << "  start-end runtime    = " << StartEndRuntime(i)/1.e6     << " ms\n";
            out << "  queue-end total time = " << FullOverheadRuntime(i)/1.e6 << " ms\n";
            
            out << endl;
        }
}
#define Safe_Round_down(n,d) Max(Round_down(n,d),0)
void fatal_CL(cl_int error, char *file, int line) {
	printf("Error in %s at line %d: ", file, line);
	
	// Print 
	switch(error) {
		case CL_SUCCESS: 									printf("CL_SUCCESS\n"); break;
		case CL_DEVICE_NOT_FOUND: 							printf("CL_DEVICE_NOT_FOUND\n"); break;
		case CL_DEVICE_NOT_AVAILABLE: 						printf("CL_DEVICE_NOT_AVAILABLE\n"); break;
		case CL_COMPILER_NOT_AVAILABLE: 					printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: 				printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
		case CL_OUT_OF_RESOURCES: 							printf("CL_OUT_OF_RESOURCES\n"); break;
		case CL_OUT_OF_HOST_MEMORY: 						printf("CL_OUT_OF_HOST_MEMORY\n"); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE: 				printf("CL_PROFILING_INFO_NOT_AVAILABLE\n"); break;
		case CL_MEM_COPY_OVERLAP: 							printf("CL_MEM_COPY_OVERLAP\n"); break;
		case CL_IMAGE_FORMAT_MISMATCH: 						printf("CL_IMAGE_FORMAT_MISMATCH\n"); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: 				printf("CL_IMAGE_FORMAT_NOT_SUPPORTED\n"); break;
		case CL_BUILD_PROGRAM_FAILURE: 						printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
		case CL_MAP_FAILURE: 								printf("CL_MAP_FAILURE\n"); break;
		case CL_INVALID_VALUE: 								printf("CL_INVALID_VALUE\n"); break;
		case CL_INVALID_DEVICE_TYPE: 						printf("CL_INVALID_DEVICE_TYPE\n"); break;
		case CL_INVALID_PLATFORM: 							printf("CL_INVALID_PLATFORM\n"); break;
		case CL_INVALID_DEVICE: 							printf("CL_INVALID_DEVICE\n"); break;
		case CL_INVALID_CONTEXT: 							printf("CL_INVALID_CONTEXT\n"); break;
		case CL_INVALID_QUEUE_PROPERTIES: 					printf("CL_INVALID_QUEUE_PROPERTIES\n"); break;
		case CL_INVALID_COMMAND_QUEUE: 						printf("CL_INVALID_COMMAND_QUEUE\n"); break;
		case CL_INVALID_HOST_PTR: 							printf("CL_INVALID_HOST_PTR\n"); break;
		case CL_INVALID_MEM_OBJECT: 						printf("CL_INVALID_MEM_OBJECT\n"); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: 			printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR\n"); break;
		case CL_INVALID_IMAGE_SIZE: 						printf("CL_INVALID_IMAGE_SIZE\n"); break;
		case CL_INVALID_SAMPLER: 							printf("CL_INVALID_SAMPLER\n"); break;
		case CL_INVALID_BINARY: 							printf("CL_INVALID_BINARY\n"); break;
		case CL_INVALID_BUILD_OPTIONS: 						printf("CL_INVALID_BUILD_OPTIONS\n"); break;
		case CL_INVALID_PROGRAM: 							printf("CL_INVALID_PROGRAM\n"); break;
		case CL_INVALID_PROGRAM_EXECUTABLE: 				printf("CL_INVALID_PROGRAM_EXECUTABLE\n"); break;
		case CL_INVALID_KERNEL_NAME: 						printf("CL_INVALID_KERNEL_NAME\n"); break;
		case CL_INVALID_KERNEL_DEFINITION: 					printf("CL_INVALID_KERNEL_DEFINITION\n"); break;
		case CL_INVALID_KERNEL: 							printf("CL_INVALID_KERNEL\n"); break;
		case CL_INVALID_ARG_INDEX: 							printf("CL_INVALID_ARG_INDEX\n"); break;
		case CL_INVALID_ARG_VALUE: 							printf("CL_INVALID_ARG_VALUE\n"); break;
		case CL_INVALID_ARG_SIZE: 							printf("CL_INVALID_ARG_SIZE\n"); break;
		case CL_INVALID_KERNEL_ARGS: 						printf("CL_INVALID_KERNEL_ARGS\n"); break;
		case CL_INVALID_WORK_DIMENSION: 					printf("CL_INVALID_WORK_DIMENSION\n"); break;
		case CL_INVALID_WORK_GROUP_SIZE: 					printf("CL_INVALID_WORK_GROUP_SIZE\n"); break;
		case CL_INVALID_WORK_ITEM_SIZE: 					printf("CL_INVALID_WORK_ITEM_SIZE\n"); break;
		case CL_INVALID_GLOBAL_OFFSET: 						printf("CL_INVALID_GLOBAL_OFFSET\n"); break;
		case CL_INVALID_EVENT_WAIT_LIST: 					printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
		case CL_INVALID_EVENT: 								printf("CL_INVALID_EVENT\n"); break;
		case CL_INVALID_OPERATION: 							printf("CL_INVALID_OPERATION\n"); break;
		case CL_INVALID_GL_OBJECT: 							printf("CL_INVALID_GL_OBJECT\n"); break;
		case CL_INVALID_BUFFER_SIZE: 						printf("CL_INVALID_BUFFER_SIZE\n"); break;
		case CL_INVALID_MIP_LEVEL: 							printf("CL_INVALID_MIP_LEVEL\n"); break;
		case CL_INVALID_GLOBAL_WORK_SIZE: 					printf("CL_INVALID_GLOBAL_WORK_SIZE\n"); break;
		
		#ifdef CL_VERSION_1_1
		case CL_MISALIGNED_SUB_BUFFER_OFFSET: 				printf("CL_MISALIGNED_SUB_BUFFER_OFFSET\n"); break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n"); break;
		case CL_INVALID_PROPERTY: 							printf("CL_INVALID_PROPERTY\n"); break;
		#endif
		
		default:											printf("Invalid OpenCL error code\n");
	}

	exit(error);
}

cl_int clBuildProgram_tony (	cl_program program,
                       const cl_device_id *device_list,
                       const char *options,
                       void (*pfn_notify)(cl_program, void *user_data),
                       void *user_data){
    return clBuildProgram(program,2,device_list,options,NULL,NULL);
    
}
cl_int clEnqueueNDRangeKernel_tony(cl_command_queue * cmdqueue,
                  cl_kernel kernel,cl_uint work_dim,
				  const size_t *global_work_size,const size_t *local_work_size
                                   ,cl_event *eventList)
  {
	//cl_event eventList[2];
     	cl_int error;
     	int CPU_RUN=0;
     	int GPU_RUN=0;
	if(tony_device==0){
		CPU_RUN=1;
		GPU_RUN=1;
	}else if(tony_device==1){
		CPU_RUN=1;
	}else{
		GPU_RUN=1;
	}
		
     size_t remain_global_work_size[2];
     size_t gpu_global_work_size[2];
     //NOTES(tony): if dim!=0, offset means offset of first dimensional.
     //we only care for dimensional that vaies.. rest Keep rest remain
         int d;
         for(d=0;d<work_dim;d++){
                 remain_global_work_size[d]=global_work_size[d];
                 gpu_global_work_size[d]=global_work_size[d];
         }
	//NOTES(tony): 
         gpu_global_work_size[0]=((double)gpu_offset/100)*(global_work_size[0]);
         gpu_global_work_size[0]=Round(gpu_global_work_size[0],(local_work_size[0]));
         remain_global_work_size[0]=global_work_size[0]-gpu_global_work_size[0];
	 if(remain_global_work_size[0]==0){
		CPU_RUN=0;
	 }
	 if(gpu_global_work_size[0]==0){
		GPU_RUN=0;
	 }
         //const size_t *const_remain = remain_global_work_size;
         printf("global_workSize[0] %d , local_work_size[1] %d, gpu_global_work_size is %d\n",global_work_size[0],local_work_size[0],gpu_global_work_size[0]);

         int numOfArgs;
         clGetKernelInfo(kernel,CL_KERNEL_NUM_ARGS,sizeof(int),&numOfArgs,NULL);
         //printf("Num of args is %d\n", numOfArgs);
         int groupOffset =0;
         if(GPU_RUN){		 
			 printf("Launch into gpu,gpu_global_work_size is %d\n",gpu_global_work_size[1]);	  
			 clSetKernelArg(kernel, numOfArgs-1, sizeof(int),(void*)&groupOffset);
			 error=clEnqueueNDRangeKernel(cmdqueue[1],kernel,work_dim,NULL,gpu_global_work_size,local_work_size,0,NULL,&(eventList[1]));
			if (error != CL_SUCCESS)
                fatal_CL(error, __FILE__,__LINE__);
         }
         if(CPU_RUN){
			 groupOffset =gpu_global_work_size[1] / (local_work_size[1]);
			 printf("Launch into cpu,global_work_size is %d\n",remain_global_work_size);	
			 clSetKernelArg(kernel, numOfArgs-1, sizeof(int),(void*)&groupOffset);
			 error =clEnqueueNDRangeKernel(cmdqueue[0],kernel,work_dim,gpu_global_work_size,remain_global_work_size,local_work_size,0,NULL,&(eventList[0]));
			if (error != CL_SUCCESS)
                fatal_CL(error, __FILE__,__LINE__);
         }
    
     int rt=-1;
     if(CPU_RUN)       error|=clFlush(cmdqueue[0]);
     if(GPU_RUN)       error|=clFlush(cmdqueue[1]);
     //  printf("Try to flush\n");
     if(CPU_RUN){
        error|=clWaitForEvents(1,&eventList[0]);
	rt+=1;
	}
     if(GPU_RUN){
        error|=clWaitForEvents(1,&eventList[1]);
	rt+=2;
	}
     //clWaitForEvents(2,eventList);
     //  printf("kernel finished\n");
     return rt;
 }

cl_mem clCreateBuffer_host(	cl_context context,cl_mem_flags flags,size_t size,
                           void *host_ptr,cl_int *errcode_ret){
    return clCreateBuffer(context,flags|CL_MEM_ALLOC_HOST_PTR,size,host_ptr,errcode_ret);
}

