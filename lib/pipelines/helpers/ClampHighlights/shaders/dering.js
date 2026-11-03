
//#region src/pipelines/helpers/ClampHighlights/shaders/dering.wgsl
var dering_default = "override DERING:bool=false;@group(1)@binding(0)var dering_stats:texture_2d<f32>;fn deringStatsAt(p:vec2i,last:vec2i)-> f32{return textureLoad(dering_stats,clamp(p,vec2i(0),last),0).x;}fn deringStats(pos:vec2f)-> f32{let dim=textureDimensions(dering_stats);let last=vec2i(dim)- 1;let t=pos*vec2f(dim)- .5;let base=floor(t);let w=t - base;let i=vec2i(base);let top=mix(deringStatsAt(i,last),deringStatsAt(i+vec2i(1,0),last),w.x);let bottom=mix(deringStatsAt(i+vec2i(0,1),last),deringStatsAt(i+vec2i(1,1),last),w.x);return mix(top,bottom,w.y);}fn deRing(color:vec4f,o:vec2i,out_dim:vec2f)-> vec4f{if(!DERING){return color;}let luma=dot(color,vec4f(.299,.587,.114,.0));let new_luma=min(luma,deringStats((vec2f(o)+.5)/out_dim));return vec4f(color.rgb -(luma - new_luma),color.a);}";
//#endregion
export { dering_default as default };
