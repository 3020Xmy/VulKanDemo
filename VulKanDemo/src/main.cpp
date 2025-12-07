// Win32 窗口与输入
#include <windows.h>
// Vulkan 核心与 Win32 表面扩展
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
// C++ 基础库
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstring>

// 从磁盘读取二进制文件（用于加载 .spv 着色器）
static std::vector<char> ReadFile(const std::wstring& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// 顶点结构：位置 (x,y,z) 与颜色 (r,g,b)
struct Vertex { float pos[3]; float color[3]; };

// 应用主结构：负责窗口、Vulkan 初始化与渲染
struct App {
    HINSTANCE hInstance{}; HWND hwnd{}; uint32_t width=800, height=600;
    VkInstance instance{}; VkSurfaceKHR surface{}; VkPhysicalDevice phys{}; uint32_t gfxQ{}, presentQ{}; VkDevice device{}; VkQueue gfx{}, present{};
    VkSwapchainKHR swapchain{}; VkFormat swapFmt{}; VkExtent2D swapExtent{}; std::vector<VkImage> swapImages; std::vector<VkImageView> swapViews;
    VkRenderPass renderPass{}; VkPipelineLayout pipeLayout{}; VkPipeline pipeline{};
    VkCommandPool cmdPool{}; std::vector<VkCommandBuffer> cmdBufs; std::vector<VkFramebuffer> framebuffers;
    VkBuffer vbuf{}; VkDeviceMemory vmem{}; VkBuffer ibuf{}; VkDeviceMemory imem{}; VkBuffer ubuf{}; VkDeviceMemory umem{};
    VkDescriptorSetLayout dsl{}; VkDescriptorPool dpool{}; VkDescriptorSet dset{};
    VkSemaphore imgAvail{}; VkSemaphore renderDone{}; std::vector<VkFence> inFlight;

    // 简单窗口过程：接收关闭消息
    static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
        if (m==WM_DESTROY) { PostQuitMessage(0); return 0; }
        return DefWindowProc(h,m,w,l);
    }

    // 创建 Win32 窗口（宽字符版本）
    void InitWindow() {
        WNDCLASSEXW wc{}; wc.cbSize=sizeof(wc); wc.style=CS_HREDRAW|CS_VREDRAW; wc.lpfnWndProc=WndProc; wc.hInstance=hInstance; wc.hCursor=LoadCursor(nullptr, IDC_ARROW); wc.lpszClassName=L"VKWin"; ::RegisterClassExW(&wc);
        RECT r{0,0,(LONG)width,(LONG)height}; AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
        hwnd=::CreateWindowExW(0,L"VKWin",L"Vulkan 3D",WS_OVERLAPPEDWINDOW|WS_VISIBLE,CW_USEDEFAULT,CW_USEDEFAULT,r.right-r.left,r.bottom-r.top,nullptr,nullptr,hInstance,nullptr);
    }

    // 创建 Vulkan 实例并启用 Win32 表面扩展
    void CreateInstance() {
        std::vector<const char*> exts{ "VK_KHR_surface","VK_KHR_win32_surface" };
        VkApplicationInfo ai{ VK_STRUCTURE_TYPE_APPLICATION_INFO }; ai.pApplicationName="VulKanDemo"; ai.apiVersion=VK_API_VERSION_1_1;
        VkInstanceCreateInfo ci{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO }; ci.pApplicationInfo=&ai; ci.enabledExtensionCount=(uint32_t)exts.size(); ci.ppEnabledExtensionNames=exts.data();
        vkCreateInstance(&ci,nullptr,&instance);
    }

    // 基于 Win32 窗口句柄创建 Vulkan 表面
    void CreateSurface() {
        VkWin32SurfaceCreateInfoKHR si{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR }; si.hinstance=hInstance; si.hwnd=hwnd; vkCreateWin32SurfaceKHR(instance,&si,nullptr,&surface);
    }

    // 选择一个同时支持图形与呈现的队列族的物理设备
    void PickDevice() {
        uint32_t c=0; vkEnumeratePhysicalDevices(instance,&c,nullptr); std::vector<VkPhysicalDevice> ds(c); vkEnumeratePhysicalDevices(instance,&c,ds.data());
        for (auto d:ds) {
            uint32_t qC=0; vkGetPhysicalDeviceQueueFamilyProperties(d,&qC,nullptr); std::vector<VkQueueFamilyProperties> qps(qC); vkGetPhysicalDeviceQueueFamilyProperties(d,&qC,qps.data());
            for (uint32_t i=0;i<qC;i++) {
                VkBool32 presentSup=VK_FALSE; vkGetPhysicalDeviceSurfaceSupportKHR(d,i,surface,&presentSup);
                if ((qps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && presentSup) { phys=d; gfxQ=i; presentQ=i; break; }
            }
            if (phys) break;
        }
    }

    // 创建逻辑设备与图形队列；启用 swapchain 扩展
    void CreateDevice() {
        float prio=1.0f; VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO }; qci.queueFamilyIndex=gfxQ; qci.queueCount=1; qci.pQueuePriorities=&prio;
        const char* exts[] = { "VK_KHR_swapchain" };
        VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO }; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=1; dci.ppEnabledExtensionNames=exts;
        vkCreateDevice(phys,&dci,nullptr,&device); vkGetDeviceQueue(device,gfxQ,0,&gfx); present=gfx;
    }

    // 创建交换链与颜色图像视图
    void CreateSwapchain() {
        uint32_t fmtCount=0; vkGetPhysicalDeviceSurfaceFormatsKHR(phys,surface,&fmtCount,nullptr); std::vector<VkSurfaceFormatKHR> fmts(fmtCount); vkGetPhysicalDeviceSurfaceFormatsKHR(phys,surface,&fmtCount,fmts.data());
        VkSurfaceFormatKHR chosen=fmts[0]; swapFmt=chosen.format;
        VkSurfaceCapabilitiesKHR caps{}; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys,surface,&caps);
        swapExtent = caps.currentExtent.width!=UINT32_MAX ? caps.currentExtent : VkExtent2D{width,height};
        uint32_t minImg = caps.minImageCount + 1; if (caps.maxImageCount>0 && minImg>caps.maxImageCount) minImg=caps.maxImageCount;
        VkSwapchainCreateInfoKHR sci{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        sci.surface=surface; sci.minImageCount=minImg; sci.imageFormat=swapFmt; sci.imageColorSpace=chosen.colorSpace; sci.imageExtent=swapExtent; sci.imageArrayLayers=1; sci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        sci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE; sci.preTransform=caps.currentTransform; sci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; sci.presentMode=VK_PRESENT_MODE_FIFO_KHR; sci.clipped=VK_TRUE;
        vkCreateSwapchainKHR(device,&sci,nullptr,&swapchain);
        uint32_t ic=0; vkGetSwapchainImagesKHR(device,swapchain,&ic,nullptr); swapImages.resize(ic); vkGetSwapchainImagesKHR(device,swapchain,&ic,swapImages.data());
        swapViews.resize(ic);
        for (uint32_t i=0;i<ic;i++) { VkImageViewCreateInfo iv{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO }; iv.image=swapImages[i]; iv.viewType=VK_IMAGE_VIEW_TYPE_2D; iv.format=swapFmt; iv.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT; iv.subresourceRange.levelCount=1; iv.subresourceRange.layerCount=1; vkCreateImageView(device,&iv,nullptr,&swapViews[i]); }
    }

    // 渲染通道：单一颜色附件，清屏并输出到呈现源
    void CreateRenderPass() {
        VkAttachmentDescription ad{}; ad.format=swapFmt; ad.samples=VK_SAMPLE_COUNT_1_BIT; ad.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; ad.storeOp=VK_ATTACHMENT_STORE_OP_STORE; ad.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; ad.finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference aref{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sd{}; sd.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS; sd.colorAttachmentCount=1; sd.pColorAttachments=&aref;
        VkSubpassDependency dep{}; dep.srcSubpass=VK_SUBPASS_EXTERNAL; dep.dstSubpass=0; dep.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; dep.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO }; rpci.attachmentCount=1; rpci.pAttachments=&ad; rpci.subpassCount=1; rpci.pSubpasses=&sd; rpci.dependencyCount=1; rpci.pDependencies=&dep; vkCreateRenderPass(device,&rpci,nullptr,&renderPass);
    }

    // 计算着色器相对 exe 的路径（从模块路径目录拼接）
    std::wstring ShaderPath(const wchar_t* name) {
        wchar_t buf[MAX_PATH]; GetModuleFileNameW(nullptr,buf,MAX_PATH); std::wstring s=buf; auto p=s.find_last_of(L"\\/"); s=s.substr(0,p+1); return s+name;
    }

    // 创建图形管线：加载 SPIR-V 着色器、顶点布局与固定功能状态
    void CreatePipeline() {
        auto vbin = ReadFile(ShaderPath(L"cube.vert.spv")); auto fbin = ReadFile(ShaderPath(L"cube.frag.spv"));
        VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO }; smci.codeSize=vbin.size(); smci.pCode=(const uint32_t*)vbin.data(); VkShaderModule vsm{}; vkCreateShaderModule(device,&smci,nullptr,&vsm);
        smci.codeSize=fbin.size(); smci.pCode=(const uint32_t*)fbin.data(); VkShaderModule fsm{}; vkCreateShaderModule(device,&smci,nullptr,&fsm);
        VkPipelineShaderStageCreateInfo vs{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO }; vs.stage=VK_SHADER_STAGE_VERTEX_BIT; vs.module=vsm; vs.pName="main";
        VkPipelineShaderStageCreateInfo fs{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO }; fs.stage=VK_SHADER_STAGE_FRAGMENT_BIT; fs.module=fsm; fs.pName="main";
        VkVertexInputBindingDescription bind{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[2]={{0,0,VK_FORMAT_R32G32B32_SFLOAT,0},{1,0,VK_FORMAT_R32G32B32_SFLOAT,12}};
        VkPipelineVertexInputStateCreateInfo vis{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO }; vis.vertexBindingDescriptionCount=1; vis.pVertexBindingDescriptions=&bind; vis.vertexAttributeDescriptionCount=2; vis.pVertexAttributeDescriptions=attrs;
        VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO }; ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkViewport vp{0,0,(float)swapExtent.width,(float)swapExtent.height,0,1}; VkRect2D sc{{0,0},swapExtent};
        VkPipelineViewportStateCreateInfo vps{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO }; vps.viewportCount=1; vps.pViewports=&vp; vps.scissorCount=1; vps.pScissors=&sc;
        // 关闭背面剔除，避免顶点绕序问题导致几何体不可见
        VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO }; rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_NONE; rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth=1.0f;
        VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO }; ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask=0xF;
        VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO }; cb.attachmentCount=1; cb.pAttachments=&cba;
        VkDescriptorSetLayoutBinding ubl{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT}; VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO }; dslci.bindingCount=1; dslci.pBindings=&ubl; vkCreateDescriptorSetLayout(device,&dslci,nullptr,&dsl);
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO }; plci.setLayoutCount=1; plci.pSetLayouts=&dsl; vkCreatePipelineLayout(device,&plci,nullptr,&pipeLayout);
        VkPipelineShaderStageCreateInfo stages[2]={vs,fs};
        VkGraphicsPipelineCreateInfo gpci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        gpci.stageCount=2; gpci.pStages=stages; gpci.pVertexInputState=&vis; gpci.pInputAssemblyState=&ia; gpci.pViewportState=&vps; gpci.pRasterizationState=&rs; gpci.pMultisampleState=&ms; gpci.pColorBlendState=&cb; gpci.layout=pipeLayout; gpci.renderPass=renderPass; gpci.subpass=0;
        vkCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&gpci,nullptr,&pipeline);
        vkDestroyShaderModule(device,vsm,nullptr); vkDestroyShaderModule(device,fsm,nullptr);
    }

    // 从物理设备内存类型中找到满足需求的类型索引
    uint32_t FindMem(uint32_t typeBits, VkMemoryPropertyFlags req) {
        VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(phys,&mp);
        for (uint32_t i=0;i<mp.memoryTypeCount;i++) if ((typeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & req)==req) return i; return 0;
    }

    // 创建缓冲并分配/绑定设备内存
    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
        VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO }; bi.size=size; bi.usage=usage; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE; vkCreateBuffer(device,&bi,nullptr,&buf);
        VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(device,buf,&mr);
        VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO }; ai.allocationSize=mr.size; ai.memoryTypeIndex=FindMem(mr.memoryTypeBits,props); vkAllocateMemory(device,&ai,nullptr,&mem); vkBindBufferMemory(device,buf,mem,0);
    }

    // 将 CPU 数据映射并写入到缓冲内存
    void Upload(VkDeviceMemory mem, const void* data, size_t size) {
        void* p=nullptr; vkMapMemory(device,mem,0,size,0,&p); memcpy(p,data,size); vkUnmapMemory(device,mem);
    }

    // 创建立方体顶点与索引数据，并上传到 GPU 可见缓冲
    void CreateGeometry() {
        std::vector<Vertex> vs={
            {{-0.5f,-0.5f,-0.5f},{1,0,0}},{{0.5f,-0.5f,-0.5f},{0,1,0}},{{0.5f,0.5f,-0.5f},{0,0,1}},{{-0.5f,0.5f,-0.5f},{1,1,0}},
            {{-0.5f,-0.5f, 0.5f},{1,0,1}},{{0.5f,-0.5f, 0.5f},{0,1,1}},{{0.5f,0.5f, 0.5f},{1,1,1}},{{-0.5f,0.5f, 0.5f},{0.5f,0.5f,0}}
        };
        std::vector<uint32_t> is={
            0,1,2,2,3,0,
            4,5,6,6,7,4,
            0,1,5,5,4,0,
            2,3,7,7,6,2,
            1,2,6,6,5,1,
            3,0,4,4,7,3
        };
        CreateBuffer(vs.size()*sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vbuf, vmem);
        Upload(vmem, vs.data(), vs.size()*sizeof(Vertex));
        CreateBuffer(is.size()*sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ibuf, imem);
        Upload(imem, is.data(), is.size()*sizeof(uint32_t));
        CreateBuffer(sizeof(float)*16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ubuf, umem);
    }

    // 创建描述符池/布局并写入 Uniform Buffer 描述
    void CreateDescriptors() {
        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1 }; VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO }; pci.maxSets=1; pci.poolSizeCount=1; pci.pPoolSizes=&ps; vkCreateDescriptorPool(device,&pci,nullptr,&dpool);
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO }; ai.descriptorPool=dpool; ai.descriptorSetCount=1; ai.pSetLayouts=&dsl; vkAllocateDescriptorSets(device,&ai,&dset);
        VkDescriptorBufferInfo bi{ ubuf,0,sizeof(float)*16 }; VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w.dstSet=dset; w.dstBinding=0; w.descriptorCount=1; w.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w.pBufferInfo=&bi; vkUpdateDescriptorSets(device,1,&w,0,nullptr);
    }

    // 基于交换链图像视图创建帧缓冲
    void CreateFramebuffers() {
        framebuffers.resize(swapViews.size());
        for (size_t i=0;i<swapViews.size();i++) { VkImageView att[]={swapViews[i]}; VkFramebufferCreateInfo fci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO }; fci.renderPass=renderPass; fci.attachmentCount=1; fci.pAttachments=att; fci.width=swapExtent.width; fci.height=swapExtent.height; fci.layers=1; vkCreateFramebuffer(device,&fci,nullptr,&framebuffers[i]); }
    }

    // 创建命令池并录制每个帧缓冲的绘制命令
    void CreateCommands() {
        VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO }; pci.queueFamilyIndex=gfxQ; vkCreateCommandPool(device,&pci,nullptr,&cmdPool);
        cmdBufs.resize(framebuffers.size()); VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO }; ai.commandPool=cmdPool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=(uint32_t)cmdBufs.size(); vkAllocateCommandBuffers(device,&ai,cmdBufs.data());
        for (size_t i=0;i<cmdBufs.size();i++) {
            VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }; vkBeginCommandBuffer(cmdBufs[i],&bi);
            VkClearValue clr{}; clr.color={{0.1f,0.1f,0.2f,1.0f}};
            VkRenderPassBeginInfo rpbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO }; rpbi.renderPass=renderPass; rpbi.framebuffer=framebuffers[i]; rpbi.renderArea.extent=swapExtent; rpbi.clearValueCount=1; rpbi.pClearValues=&clr; vkCmdBeginRenderPass(cmdBufs[i],&rpbi,VK_SUBPASS_CONTENTS_INLINE);
            VkDeviceSize offs=0; vkCmdBindPipeline(cmdBufs[i],VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline); vkCmdBindVertexBuffers(cmdBufs[i],0,1,&vbuf,&offs); vkCmdBindIndexBuffer(cmdBufs[i],ibuf,0,VK_INDEX_TYPE_UINT32); vkCmdBindDescriptorSets(cmdBufs[i],VK_PIPELINE_BIND_POINT_GRAPHICS,pipeLayout,0,1,&dset,0,nullptr);
            vkCmdDrawIndexed(cmdBufs[i],6,1,0,0,0);
            vkCmdEndRenderPass(cmdBufs[i]); vkEndCommandBuffer(cmdBufs[i]);
        }
    }

    // 创建同步对象：两信号量 + 每帧一个栅栏
    void CreateSync() {
        VkSemaphoreCreateInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }; vkCreateSemaphore(device,&si,nullptr,&imgAvail); vkCreateSemaphore(device,&si,nullptr,&renderDone);
        inFlight.resize(framebuffers.size()); VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; fi.flags=VK_FENCE_CREATE_SIGNALED_BIT; for (auto& f:inFlight) vkCreateFence(device,&fi,nullptr,&f);
    }

    // 每帧更新模型矩阵（绕 Y 轴旋转 + 缩放），上传到 Uniform Buffer
    void UpdateUniform(float t) {
        float c=std::cos(t), s=std::sin(t); float k=0.7f;
        float model[16]={
            k*c, 0, -k*s, 0,
            0,   k,   0,  0,
            k*s, 0,  k*c, 0,
            0,   0,   0,  1
        };
        Upload(umem, model, sizeof(model));
    }

    // 获取交换链图像 → 提交绘制命令 → 呈现到屏幕
    void DrawFrame() {
        uint32_t idx=0; vkAcquireNextImageKHR(device,swapchain,UINT64_MAX,imgAvail,VK_NULL_HANDLE,&idx);
        VkPipelineStageFlags waitStage=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.waitSemaphoreCount=1; si.pWaitSemaphores=&imgAvail; si.pWaitDstStageMask=&waitStage; si.commandBufferCount=1; si.pCommandBuffers=&cmdBufs[idx]; si.signalSemaphoreCount=1; si.pSignalSemaphores=&renderDone;
        vkWaitForFences(device,1,&inFlight[idx],VK_TRUE,UINT64_MAX); vkResetFences(device,1,&inFlight[idx]); vkQueueSubmit(gfx,1,&si,inFlight[idx]);
        VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR }; pi.waitSemaphoreCount=1; pi.pWaitSemaphores=&renderDone; pi.swapchainCount=1; pi.pSwapchains=&swapchain; pi.pImageIndices=&idx; vkQueuePresentKHR(present,&pi);
    }

    // 初始化 → 主循环：更新 + 绘制 → 等待设备空闲
    void Run() {
        InitWindow(); CreateInstance(); CreateSurface(); PickDevice(); CreateDevice(); CreateSwapchain(); CreateRenderPass(); CreatePipeline(); CreateGeometry(); CreateDescriptors(); CreateFramebuffers(); CreateCommands(); CreateSync();
        auto start=std::chrono::steady_clock::now();
        MSG msg{}; while (msg.message!=WM_QUIT) { if (PeekMessage(&msg,nullptr,0,0,PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); } else { auto now=std::chrono::steady_clock::now(); float t=std::chrono::duration<float>(now-start).count(); UpdateUniform(t); DrawFrame(); } }
        vkDeviceWaitIdle(device);
    }
};

// Windows GUI 入口：创建并运行应用
int WINAPI wWinMain(HINSTANCE hInst,HINSTANCE,LPWSTR,int) {
    App app; app.hInstance=hInst; app.Run(); return 0;
}
