
//#region src/pipelines/helpers/FrameGate/shaders/copy.wgsl
var copy_default = "@group(0)@binding(0)var tex_cur:texture_2d<f32>;@group(0)@binding(1)var tex_prev:texture_storage_2d<rgba8unorm,write>;@compute @workgroup_size(8,8)fn computeMain(@builtin(global_invocation_id)pixel:vec3u){let dim:vec2u=textureDimensions(tex_cur);if(pixel.x >=dim.x || pixel.y >=dim.y){return;}textureStore(tex_prev,pixel.xy,textureLoad(tex_cur,pixel.xy,0));}";
//#endregion
export { copy_default as default };
