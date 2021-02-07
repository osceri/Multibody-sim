//??GL_VERTEX_SHADER
#version 450 core

layout(location = 0) in vec3 normals;
layout(location = 1) in vec2 uvs;

uniform mat4 mv;
uniform mat4 p;

out vec4 position;
out vec4 normal;
out vec2 uv;

void main() {
	position = mv * vec4(normals, 1.);
	normal = mv * vec4(normals, 0.);
	uv = uvs;

	gl_Position = p * position;
}

//??GL_FRAGMENT_SHADER
#version 450 core

out vec4 color;

uniform float time;

in vec4 position;
in vec4 normal;
in vec2 uv;

void main() {
	float f = 0.8 - 0.2 * dot(normalize(normal.xyz), normalize(position.xyz));

	color = vec4(f*vec3(0.74, 0.66, 0.55), 1.);

}