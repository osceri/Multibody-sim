//??GL_VERTEX_SHADER
#version 450 core

layout(location = 0) in vec3 normals;
layout(location = 1) in vec2 uvs;

uniform mat4 mv;
uniform mat4 p;

out vec4 position;
out vec4 normal;

void main() {
	position = mv * vec4(normals, 1.);
	normal = mv * vec4(normals, 0.);

	gl_Position = p * position;
}

//??GL_FRAGMENT_SHADER
#version 450 core

out vec4 color;


uniform float time; 
uniform vec4 light;


in vec4 position;
in vec4 normal;

void main() {
	float f = 0.5 + 0.5 * dot(normalize(normal.xyz), normalize(light.xyz - position.xyz));
	
	color = vec4(f*vec3(0.34, 0.36, 0.45), 1.);

}