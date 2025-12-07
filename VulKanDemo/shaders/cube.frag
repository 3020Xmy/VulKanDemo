#version 450
// 片段着色器：输出插值后的颜色
layout(location=0) in vec3 vColor;        // 从顶点阶段传来的颜色
layout(location=0) out vec4 outColor;     // 写入帧缓冲的颜色
void main(){
  outColor = vec4(vColor,1.0);            // 固定 alpha=1
}
