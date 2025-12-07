#version 450
// 顶点着色器：输入位置/颜色，乘以矩阵后传递颜色到片段
layout(location=0) in vec3 inPos;     // 顶点位置
layout(location=1) in vec3 inColor;   // 顶点颜色
layout(binding=0) uniform UB { mat4 mvp; } ub; // 统一缓冲中的 4x4 矩阵
layout(location=0) out vec3 vColor;   // 传递到片段的颜色
void main(){
  gl_Position = ub.mvp * vec4(inPos,1.0); // 变换到裁剪空间
  vColor = inColor;                        // 保留原始颜色
}
